"""Diagnostics for data-free parameter saliency and path connectivity using SynFlow."""

import torch

from syndset.utils import extract_primary_tensor

_extract_primary_tensor = extract_primary_tensor


def check_synflow(model, sample_input):
  """Computes data-free synaptic flow to detect structural bottlenecks and starved layers.

  Theoretical background:
    SynFlow (Tanaka et al., NeurIPS 2020) evaluates the conservation of synaptic flow
    through a network without data. It pushes an all-ones input through absolute parameter
    weights theta' = |theta| and evaluates the metric:
      R = sum(f(1; |theta|))
    The gradient-parameter product S(theta) = sum |theta' * grad_{theta'} R| quantifies
    how effectively information flows through every parameterized submodule.

    Layers with S(theta) = 0 cannot transmit signals to the loss, indicating structural
    disconnection, unreached residual paths, or dead branches. Starvation evaluation
    is scoped to multi-dimensional weight tensors (dim >= 2), excluding 1D affine
    biases and normalization offsets which do not represent synaptic paths and
    naturally exhibit zero data-free flow when zero-initialized.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: Sample input tensor matching model forward signature.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'total_synflow': Total scalar synaptic flow through the model.
      - 'layer_synflow': Dict mapping parameter names to their SynFlow score.
      - 'starved_layers': List of parameter names with zero or negligible flow.
      - 'issues': Human-readable diagnostic recommendations.
  """
  model.eval()

  # Create an all-ones input tensor
  if isinstance(sample_input, torch.Tensor):
    if sample_input.is_floating_point():
      ones_input = torch.ones_like(sample_input)
    else:
      # Discrete token IDs
      ones_input = torch.ones_like(sample_input)
  elif isinstance(sample_input, (tuple, list)):
    ones_input = [torch.ones_like(x) if isinstance(x, torch.Tensor) else x for x in sample_input]
  elif isinstance(sample_input, dict):
    ones_input = {
      k: torch.ones_like(v) if isinstance(v, torch.Tensor) else v for k, v in sample_input.items()
    }
  else:
    ones_input = sample_input

  # Store original parameter values and replace with abs(param)
  orig_params = {}
  for name, param in model.named_parameters():
    if param.requires_grad:
      orig_params[name] = param.data.clone()
      param.data = param.data.abs()
      param.grad = None

  try:
    if isinstance(ones_input, dict):
      output = model(**ones_input)
    elif isinstance(ones_input, (tuple, list)):
      output = model(*ones_input)
    else:
      output = model(ones_input)

    tensor_out = _extract_primary_tensor(output)
    if tensor_out is None or tensor_out.numel() == 0:
      return {
        "status": "failed",
        "total_synflow": 0.0,
        "layer_synflow": {},
        "starved_layers": list(orig_params.keys()),
        "issues": ["Model output does not contain any floating-point tensor for backpropagation."],
      }

    metric = tensor_out.sum()
    metric.backward()

    layer_synflow = {}
    starved_layers = []
    total_synflow = 0.0

    for name, param in model.named_parameters():
      if not param.requires_grad:
        continue
      if param.grad is not None:
        flow = (param.data * param.grad.abs()).sum().item()
      else:
        flow = 0.0

      layer_synflow[name] = float(flow)
      total_synflow += flow

    # SynFlow evaluates conservation of flow through multi-dimensional synaptic weight
    # tensors (dim >= 2). 1D affine biases and normalization scales/shifts do not form
    # synaptic connections between layers and often carry zero data-free flow by design.
    synaptic_names = [
      name for name, p in model.named_parameters() if p.requires_grad and p.dim() >= 2
    ]
    eval_names = synaptic_names if synaptic_names else list(layer_synflow.keys())

    for name in eval_names:
      if layer_synflow.get(name, 0.0) <= 1e-12:
        starved_layers.append(name)

  finally:
    # Restore original parameter values and clear grads
    for name, orig in orig_params.items():
      param = dict(model.named_parameters())[name]
      param.data = orig
      param.grad = None

  issues = []
  if total_synflow <= 1e-12:
    status = "failed"
    issues.append(
      "Total synaptic flow is 0.0! Network is completely disconnected or outputs constant zero.")
  elif len(starved_layers) > 0:
    fraction_starved = len(starved_layers) / max(1, len(eval_names))
    if fraction_starved > 0.3:
      status = "failed"
      issues.append(f"{len(starved_layers)} of {len(eval_names)} synaptic weight tensors have zero "
                    f"synaptic flow ({fraction_starved:.1%}).")
    else:
      status = "warning"
      issues.append(f"{len(starved_layers)} synaptic weight tensors have zero synaptic flow "
                    f"(e.g. '{starved_layers[0]}').")
  else:
    status = "healthy"

  return {
    "status": status,
    "total_synflow": float(total_synflow),
    "layer_synflow": layer_synflow,
    "starved_layers": starved_layers,
    "issues": issues,
  }
