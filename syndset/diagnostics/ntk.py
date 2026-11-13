"""Diagnostics for evaluating empirical Neural Tangent Kernel (NTK) conditioning."""

import torch

from syndset.utils import extract_primary_tensor

_extract_primary_tensor = extract_primary_tensor


def check_ntk_conditioning(
  model,
  sample_input,
  sample_target=None,
  max_samples=8,
  condition_threshold=1e5,
):
  """Computes empirical NTK condition number to evaluate convergence uniformity.

  Theoretical background:
    In the infinite-width limit (Jacot et al., NeurIPS 2018), neural network training
    under continuous gradient flow is governed by the Neural Tangent Kernel:
      Theta_{i,j} = < grad_theta f(x_i), grad_theta f(x_j) >
    The condition number kappa(Theta) = lambda_max / lambda_min determines optimization speed:
    - Low condition number: All error components converge at comparable, fast rates.
    - Extreme condition number (kappa >> 10^5): Severe stiffness, saddle point stagnation,
      and poor generalization on complex modes.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: Sample tensor with a batch dimension (B, ...).
    sample_target: Optional target tensor, accepted for signature uniformity across probes.
    max_samples: Number of sample rows N to use for empirical NTK (default: 8).
    condition_threshold: Condition number threshold triggering warning/failure (default: 1e5).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'condition_number': Ratio of lambda_max to lambda_min.
      - 'eigenvalues': List of NTK eigenvalues sorted in descending order.
      - 'trace': Trace of the empirical NTK matrix.
      - 'issues': Actionable diagnostic recommendations.
  """
  # Support positional max_samples for backward compatibility if passed as 3rd argument
  if isinstance(sample_target, int):
    max_samples = sample_target
    sample_target = None

  model.eval()

  if not isinstance(sample_input, torch.Tensor):
    return {
      "status": "healthy",
      "condition_number": 1.0,
      "eigenvalues": [1.0],
      "trace": 1.0,
      "issues": ["Sample input is not a single Tensor; NTK conditioning skipped."],
    }

  batch_size = sample_input.shape[0]
  n_samples = min(batch_size, max_samples)
  if n_samples < 2:
    return {
      "status": "healthy",
      "condition_number": 1.0,
      "eigenvalues": [1.0],
      "trace": 1.0,
      "issues": ["Fewer than 2 samples provided; NTK matrix requires at least 2 samples."],
    }

  # Ensure model parameters have requires_grad
  param_list = [p for p in model.parameters() if p.requires_grad]
  if not param_list:
    return {
      "status": "failed",
      "condition_number": float("inf"),
      "eigenvalues": [],
      "trace": 0.0,
      "issues": ["Model has no parameters with requires_grad=True."],
    }

  sample_grads = []

  for i in range(n_samples):
    single_input = sample_input[i:i + 1]
    model.zero_grad(set_to_none=True)

    out = model(single_input)
    tensor_out = _extract_primary_tensor(out)
    if tensor_out is None:
      return {
        "status": "failed",
        "condition_number": float("inf"),
        "eigenvalues": [],
        "trace": 0.0,
        "issues": ["Could not extract primary tensor from model output."],
      }

    scalar_loss = tensor_out.sum()
    grads = torch.autograd.grad(
      scalar_loss,
      param_list,
      retain_graph=False,
      create_graph=False,
      allow_unused=True,
    )

    flat_g = torch.cat([
      g.reshape(-1) if g is not None else torch.zeros(p.numel(), device=p.device)
      for g, p in zip(grads, param_list)
    ])
    sample_grads.append(flat_g.detach())

  model.zero_grad(set_to_none=True)

  # Form G of shape (N, P) and NTK = G @ G.T of shape (N, N)
  grad_matrix = torch.stack(sample_grads, dim=0).float()
  ntk_matrix = torch.matmul(grad_matrix, grad_matrix.t())

  # SVD / Eigenvalues of symmetric positive semi-definite matrix
  eigenvalues = torch.linalg.eigvalsh(ntk_matrix).cpu().tolist()
  eigenvalues.sort(reverse=True)

  lambda_max = eigenvalues[0] if eigenvalues else 0.0
  lambda_min = eigenvalues[-1] if eigenvalues else 0.0
  trace_val = sum(eigenvalues)

  eps = 1e-12
  if lambda_min <= eps:
    condition_number = float("inf")
  else:
    condition_number = float(lambda_max / (lambda_min + eps))

  issues = []
  if lambda_max <= eps:
    status = "failed"
    issues.append("NTK has zero trace! Gradients are completely vanishing for all probe samples.")
  elif condition_number > condition_threshold:
    status = "warning" if condition_number < 1e7 else "failed"
    issues.append(f"Empirical NTK is severely ill-conditioned "
                  f"(kappa={condition_number:.2e} > {condition_threshold:.0e}). "
                  f"High-frequency error modes will converge orders of magnitude slower.")
  else:
    status = "healthy"

  return {
    "status": status,
    "condition_number": condition_number,
    "eigenvalues": eigenvalues,
    "trace": float(trace_val),
    "issues": issues,
  }
