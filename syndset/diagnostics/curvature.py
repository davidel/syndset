"""Diagnostics for evaluating loss surface curvature and sharpness."""

import torch
import torch.nn as nn


def _extract_primary_tensor(output):
  """Extracts the primary floating-point tensor from model return structures.

  Args:
    output: Model output, which could be a Tensor, tuple/list, or object with .logits.

  Returns:
    A torch.Tensor if found, or None.
  """
  if isinstance(output, torch.Tensor):
    return output
  if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
    return output[0]
  if hasattr(output, "logits"):
    return output.logits
  return None


def check_curvature_sharpness(model, sample_input, target=None, loss_fn=None, num_samples=5):
  """Estimates loss landscape sharpness and Hessian trace using Hutchinson's method.

  Theoretical background:
    The Hessian matrix H = grad^2_theta L in R^{P x P} describes the local curvature
    of the loss surface around parameters theta. Sharp minima (high Tr(H) or large
    lambda_max) generalize poorly and are fragile under quantization and mixed precision.

    Because computing the full P x P Hessian is computationally intractable for deep models,
    this probe uses Hutchinson's randomized trace estimator (Hutchinson, 1989):
      Tr(H) = E_{v ~ N(0, I)} [ v^T H v ]
            = (1 / M) sum_{m=1}^M v_m^T grad_theta (grad_theta L^T v_m)
    which evaluates the trace in M fast vector-Hessian products without ever forming H.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: Sample input tensor matching model forward signature.
    target: Optional ground-truth target tensor.
    loss_fn: Optional loss callable. If None, defaults to output.sum().
    num_samples: Number of random vector-Hessian probes M (default: 5).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'hessian_trace': Estimated total trace of the Hessian Tr(H).
      - 'mean_curvature': Mean eigenvalue estimate Tr(H) / total_params.
      - 'issues': Human-readable diagnostic recommendations.
  """
  model.eval()
  model.zero_grad(set_to_none=True)

  # 1. Forward pass
  if isinstance(sample_input, dict):
    output = model(**sample_input)
  elif isinstance(sample_input, (tuple, list)):
    output = model(*sample_input)
  else:
    output = model(sample_input)

  pred = _extract_primary_tensor(output)

  # 2. Compute loss
  if loss_fn is not None:
    loss = loss_fn(pred if pred is not None else output, target)
  elif target is not None:
    if pred is None:
      raise ValueError("Could not extract tensor from model output to compute loss.")
    if target.dtype in (torch.long, torch.int64):
      default_loss = nn.CrossEntropyLoss()
      loss = default_loss(pred, target)
    else:
      default_loss = nn.MSELoss()
      loss = default_loss(pred, target)
  else:
    if pred is not None:
      loss = pred.sum()
    else:
      raise ValueError("Could not compute scalar loss from model output.")

  # 3. Trainable parameters
  params = [p for p in model.parameters() if p.requires_grad]
  total_params = sum(p.numel() for p in params)
  if total_params == 0:
    return {
      "status": "warning",
      "hessian_trace": 0.0,
      "mean_curvature": 0.0,
      "issues": ["Model contains no trainable parameters."],
    }

  # 4. First backward pass with graph retention
  grads = torch.autograd.grad(loss, params, create_graph=True)

  trace_estimates = []

  # 5. Hutchinson randomized projections
  for idx in range(num_samples):
    # Sample Rademacher random vectors {-1, +1}
    v_list = [torch.randint(0, 2, p.shape, device=p.device).float() * 2.0 - 1.0 for p in params]

    # Inner product: sum_p (grad_p * v_p)
    inner_prod = sum((g * v).sum() for g, v in zip(grads, v_list))

    # Second backward pass: Hv = grad_theta (g^T v)
    retain = idx < num_samples - 1
    hv_list = torch.autograd.grad(inner_prod, params, retain_graph=retain)

    # Quadratic form: v^T (Hv)
    quad_form = sum((v * hv).sum().item() for v, hv in zip(v_list, hv_list))
    trace_estimates.append(quad_form)

  mean_trace = sum(trace_estimates) / max(len(trace_estimates), 1)
  mean_curvature = mean_trace / max(total_params, 1)

  issues = []
  if mean_curvature > 50.0:
    issues.append(
      f"High loss curvature detected (mean eigenvalue estimate: {mean_curvature:.2f}). "
      "The loss landscape is sharp, which may cause training instability and poor generalization."
    )
  elif mean_curvature < -1.0:
    issues.append(
      f"Negative curvature detected (mean eigenvalue estimate: {mean_curvature:.2f}). "
      "Current initialization lies near a concave ridge or saddle point."
    )

  if mean_curvature > 200.0:
    status = "warning"
  elif issues:
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "hessian_trace": float(mean_trace),
    "mean_curvature": float(mean_curvature),
    "issues": issues,
  }
