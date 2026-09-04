"""Diagnostics for evaluating permutation equivariance and invariance in set/graph models."""

import torch


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


def check_permutation_equivariance(model,
                                   sample_input,
                                   perm_dim=1,
                                   check_invariance=False,
                                   rtol=1e-3,
                                   seed=42):
  """Tests whether an architecture exhibits permutation equivariance or invariance.

  Theoretical background:
    In set architectures (DeepSets, Set Transformers) and Graph Neural Networks (GNNs),
    the order of elements or nodes is arbitrary. The model must satisfy either:
      - Permutation Equivariance: f(pi(X)) = pi(f(X)) for node/token representations.
      - Permutation Invariance:   f(pi(X)) = f(X)      for graph/set-level predictions.
    This probe applies a random coordinate permutation pi along perm_dim and measures
    the relative Frobenius norm difference between expected and observed outputs:
      rel_diff = ||f(pi(X)) - expected||_F / (||expected||_F + 1e-7)

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: A torch.Tensor input to permute.
    perm_dim: The dimension along which to apply the permutation (default: 1).
    check_invariance: If True, tests invariance f(pi(X)) == f(X). If False,
      tests equivariance f(pi(X)) == pi(f(X)) (default: False).
    rtol: Relative tolerance threshold to consider the property satisfied (default: 1e-3).
    seed: Random seed for reproducible permutation (default: 42).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy' or 'warning'.
      - 'is_satisfied': True if the relative difference is below rtol.
      - 'relative_difference': Float relative Frobenius norm error.
      - 'property_tested': String ('equivariance' or 'invariance').
      - 'issues': Human-readable diagnostic recommendations.
  """
  model.eval()

  if not isinstance(sample_input, torch.Tensor):
    raise TypeError("sample_input must be a torch.Tensor for permutation testing.")

  if perm_dim >= sample_input.dim():
    raise ValueError(f"perm_dim ({perm_dim}) exceeds tensor dimensions ({sample_input.dim()}).")

  dim_size = sample_input.shape[perm_dim]
  generator = torch.Generator().manual_seed(seed)
  perm = torch.randperm(dim_size, generator=generator)

  # Permute input along perm_dim
  perm_input = torch.index_select(sample_input, perm_dim, perm)

  with torch.no_grad():
    y_orig_raw = model(sample_input)
    y_perm_raw = model(perm_input)

  y_orig = _extract_primary_tensor(y_orig_raw)
  y_perm = _extract_primary_tensor(y_perm_raw)

  if y_orig is None or y_perm is None:
    return {
      "status": "warning",
      "is_satisfied": False,
      "relative_difference": float("inf"),
      "property_tested": "invariance" if check_invariance else "equivariance",
      "issues": ["Could not extract tensor from model output to verify permutation symmetry."],
    }

  if check_invariance:
    # Invariance expects f(pi(X)) == f(X)
    y_expected = y_orig
    property_name = "invariance"
  else:
    # Equivariance expects f(pi(X)) == pi(f(X))
    if perm_dim < y_orig.dim() and y_orig.shape[perm_dim] == dim_size:
      y_expected = torch.index_select(y_orig, perm_dim, perm)
    else:
      y_expected = y_orig
    property_name = "equivariance"

  diff_norm = torch.norm(y_perm.float() - y_expected.float()).item()
  base_norm = torch.norm(y_expected.float()).item()
  rel_diff = diff_norm / (base_norm + 1e-7)

  is_satisfied = bool(rel_diff <= rtol)
  issues = []

  if not is_satisfied:
    issues.append(
      f"Model does not satisfy permutation {property_name} along dimension {perm_dim} "
      f"(relative error: {rel_diff:.2%}). Check if positional encodings or asymmetric operations "
      "break symmetry.")
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "is_satisfied": is_satisfied,
    "relative_difference": float(rel_diff),
    "property_tested": property_name,
    "issues": issues,
  }
