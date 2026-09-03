"""Numerical and statistical helper functions for model health checks."""

import torch


def compute_tensor_norm(tensor, p=2):
  """Computes the p-norm of a tensor.

  Args:
    tensor: A torch.Tensor.
    p: The order of norm (default: 2).

  Returns:
    A float value of the computed norm. Returns 0.0 if tensor has no elements.
  """
  if tensor is None or tensor.numel() == 0:
    return 0.0
  norm_val = torch.norm(tensor.float(), p=p).item()
  return float(norm_val)


def compute_effective_rank(matrix, eps=1e-10):
  """Computes the effective rank of a 2D activation matrix using spectral entropy.

  The effective rank measures how many dimensions of the representation space
  are genuinely utilized, based on the Roy and Vetterli (2007) definition:
  erank(A) = exp(H(p)), where p_i = sigma_i / sum(sigma) and H is Shannon entropy.

  Args:
    matrix: A 2D torch.Tensor of shape (samples, features).
    eps: Small float to prevent log(0).

  Returns:
    A float between 1.0 and min(samples, features). Returns 0.0 if input is
    degenerate or has fewer than 2 dimensions.
  """
  if matrix.dim() != 2 or matrix.numel() == 0:
    return 0.0
  matrix_float = matrix.float()
  try:
    singular_values = torch.linalg.svdvals(matrix_float)
  except RuntimeError:
    return 0.0
  total_energy = torch.sum(singular_values)
  if total_energy <= eps:
    return 0.0
  probabilities = singular_values / total_energy
  entropy = -torch.sum(probabilities * torch.log(probabilities + eps))
  effective_rank = torch.exp(entropy).item()
  return float(effective_rank)


def summarize_tensor(tensor):
  """Generates a summary dictionary of values in a tensor.

  Args:
    tensor: A torch.Tensor to inspect.

  Returns:
    A dictionary containing norm, min, max, mean, std, zero_fraction,
    has_nan, and has_inf.
  """
  if tensor is None:
    return {
      "norm_l2": 0.0,
      "norm_linf": 0.0,
      "min": 0.0,
      "max": 0.0,
      "mean": 0.0,
      "std": 0.0,
      "zero_fraction": 1.0,
      "has_nan": False,
      "has_inf": False,
    }
  flat = tensor.float().flatten()
  has_nan = bool(torch.isnan(flat).any().item())
  has_inf = bool(torch.isinf(flat).any().item())
  if has_nan or has_inf:
    return {
      "norm_l2": float("nan"),
      "norm_linf": float("nan"),
      "min": float("nan"),
      "max": float("nan"),
      "mean": float("nan"),
      "std": float("nan"),
      "zero_fraction": 0.0,
      "has_nan": has_nan,
      "has_inf": has_inf,
    }
  norm_l2 = float(torch.norm(flat, 2).item())
  norm_linf = float(torch.norm(flat, float("inf")).item())
  min_val = float(torch.min(flat).item())
  max_val = float(torch.max(flat).item())
  mean_val = float(torch.mean(flat).item())
  std_val = float(torch.std(flat).item()) if flat.numel() > 1 else 0.0
  zero_fraction = float((flat == 0).sum().item() / flat.numel())
  return {
    "norm_l2": norm_l2,
    "norm_linf": norm_linf,
    "min": min_val,
    "max": max_val,
    "mean": mean_val,
    "std": std_val,
    "zero_fraction": zero_fraction,
    "has_nan": has_nan,
    "has_inf": has_inf,
  }
