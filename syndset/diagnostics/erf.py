"""Diagnostics for measuring Effective Receptive Field (ERF) spread and artifacts."""

import torch

from syndset.utils import extract_primary_tensor

_extract_primary_tensor = extract_primary_tensor


def check_effective_receptive_field(
    model,
    input_shape=(1, 3, 32, 32),
    min_radius_ratio=0.15,
    max_checkerboard_ratio=0.6,
):
  """Computes Effective Receptive Field (ERF) dispersion and spatial grid artifacts.

  Theoretical background:
    The Effective Receptive Field (Luo et al., NeurIPS 2016) measures the region of input
    pixels that significantly influence a central output feature. It follows a 2D Gaussian
    profile:
      ERF(i, j) = | d(y_center) / d(x_ij) |
    Architectures with tiny ERF radius ratios cannot model global context. High checkerboard
    energy ratios indicate strided deconvolution artifacts that cause visual hallucination.

  Args:
    model: The vision torch.nn.Module to evaluate. Accepts 4D tensor (B, C, H, W).
    input_shape: 4-tuple input dimensions (batch, channels, height, width).
    min_radius_ratio: Minimum acceptable ERF radius normalized by spatial dimension.
    max_checkerboard_ratio: Maximum acceptable high-frequency checkerboard ratio.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'erf_radius_ratio': Estimated 2-sigma radius divided by spatial dimension.
      - 'checkerboard_ratio': Fraction of high-frequency alternating energy.
      - 'issues': Actionable diagnostic recommendations.
  """
  model.eval()

  # Create an input tensor with requires_grad=True
  x = torch.randn(*input_shape, requires_grad=True)
  model.zero_grad(set_to_none=True)

  out = model(x)
  feature_map = _extract_primary_tensor(out)

  if feature_map is None:
    return {
      "status": "failed",
      "erf_radius_ratio": 0.0,
      "checkerboard_ratio": 1.0,
      "issues": ["Could not extract primary tensor from model output."],
    }

  # Pick central coordinate of feature map
  if feature_map.dim() >= 2:
    center_idx = [0] + [s // 2 for s in feature_map.shape[1:]]
    center_target = feature_map[tuple(center_idx)]
  else:
    center_target = feature_map.sum()

  grads = torch.autograd.grad(center_target.sum(), x, retain_graph=False)[0]
  if grads is None or grads.numel() == 0:
    return {
      "status": "failed",
      "erf_radius_ratio": 0.0,
      "checkerboard_ratio": 1.0,
      "issues": ["Gradient with respect to input is None."],
    }

  # Average across channels and batch: shape (H, W)
  erf_map = grads.abs().mean(dim=(0, 1)).detach()
  h, w = erf_map.shape
  total_mass = erf_map.sum()

  if total_mass <= 1e-12:
    return {
      "status": "failed",
      "erf_radius_ratio": 0.0,
      "checkerboard_ratio": 1.0,
      "issues": ["ERF gradient map is completely zero."],
    }

  # Normalize to probability distribution
  prob_map = erf_map / total_mass

  # Compute center of mass
  grid_y, grid_x = torch.meshgrid(
    torch.arange(h, dtype=torch.float32),
    torch.arange(w, dtype=torch.float32),
    indexing="ij",
  )
  mean_y = (prob_map * grid_y).sum()
  mean_x = (prob_map * grid_x).sum()

  var_y = (prob_map * (grid_y - mean_y)**2).sum()
  var_x = (prob_map * (grid_x - mean_x)**2).sum()

  sigma = torch.sqrt(torch.clamp(var_y + var_x, min=1e-8)).item()
  radius_ratio = float((2.0 * sigma) / max(h, w))

  # Checkerboard / high-frequency metric: ratio of neighboring difference energy
  diff_y = (erf_map[1:, :] - erf_map[:-1, :]).abs().sum()
  diff_x = (erf_map[:, 1:] - erf_map[:, :-1]).abs().sum()
  neighbor_diff_energy = (diff_y + diff_x) / (total_mass + 1e-12)
  checkerboard_ratio = float(min(1.0, neighbor_diff_energy.item() / 2.0))

  issues = []
  if radius_ratio < min_radius_ratio:
    status = "warning"
    issues.append(f"Constricted ERF radius ratio ({radius_ratio:.1%} < {min_radius_ratio:.1%}). "
                  f"Model has insufficient global context reception.")
  elif checkerboard_ratio > max_checkerboard_ratio:
    status = "warning"
    issues.append(
      f"High spatial checkerboard noise ({checkerboard_ratio:.1%} > {max_checkerboard_ratio:.1%}). "
      f"Indicates strided/deconv aliasing artifacts.")
  else:
    status = "healthy"

  return {
    "status": status,
    "erf_radius_ratio": radius_ratio,
    "checkerboard_ratio": checkerboard_ratio,
    "issues": issues,
  }
