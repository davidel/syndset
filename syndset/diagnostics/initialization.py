"""Diagnostics for checking parameter initialization scaling and spectral radius."""

import math

import torch


def check_initialization_scale(model, scale_tolerance_ratio=10.0):
  """Evaluates whether weight tensors follow stable initialization envelopes.

  Theoretical background:
    Improper weight initialization is the root cause of early gradient failure:
    - Kaiming Normal (He et al., 2015): sigma = sqrt(2 / fan_in).
    - Xavier / Glorot (Glorot & Bengio, 2010): sigma = sqrt(2 / (fan_in + fan_out)).
    If weights are initialized too large (sigma_emp >> sigma_theo), activations and logits
    explode into saturation or overflow FP16. If initialized too small, signal decays.

    For square linear/recurrent weight matrices W in R^{d x d}, the spectral radius:
      rho(W) = max_i |lambda_i(W)|
    governs dynamical stability. In recurrent and deep residual paths, rho(W) > 1.0 causes
    exponential growth across time/depth, while rho(W) << 1.0 causes exponential decay.

  Args:
    model: The torch.nn.Module to evaluate.
    scale_tolerance_ratio: Ratio above or below theoretical variance that triggers
      a warning (default: 10.0).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'layer_stats': Dict of parameter statistics (mean, std, fan_in, ratio).
      - 'spectral_radii': Dict of spectral radius values for square 2D weight matrices.
      - 'issues': Human-readable diagnostic recommendations.
  """
  layer_stats = {}
  spectral_radii = {}
  issues = []

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue

    data = param.data.float()
    numel = data.numel()

    # Check for NaN / Inf at initialization
    if torch.isnan(data).any() or torch.isinf(data).any():
      issues.append(f"Parameter '{name}' contains NaN or Inf values at initialization!")
      continue

    # Only evaluate multidimensional weight tensors (dim >= 2)
    if data.dim() >= 2:
      fan_in = data.shape[1]
      fan_out = data.shape[0]
      if data.dim() > 2:
        receptive_field_size = math.prod(data.shape[2:])
        fan_in *= receptive_field_size
        fan_out *= receptive_field_size

      emp_mean = float(data.mean().item())
      emp_std = float(data.std().item()) if numel > 1 else 0.0

      # Theoretical Kaiming std
      theoretical_std = math.sqrt(2.0 / max(fan_in, 1))
      scale_ratio = emp_std / max(theoretical_std, 1e-12)

      layer_stats[name] = {
        "mean": emp_mean,
        "std": emp_std,
        "fan_in": fan_in,
        "fan_out": fan_out,
        "theoretical_std": theoretical_std,
        "scale_ratio": scale_ratio,
      }

      if emp_std == 0.0:
        issues.append(f"Weight '{name}' has zero variance (all weights identical or zero).")
      elif scale_ratio > scale_tolerance_ratio:
        issues.append(f"Weight '{name}' has abnormally large scale: empirical std {emp_std:.4f} "
                      f"is {scale_ratio:.1f}x theoretical Kaiming std ({theoretical_std:.4f}). "
                      "Risk of logit explosion and FP16 overflow.")
      elif scale_ratio < (1.0 / scale_tolerance_ratio):
        issues.append(f"Weight '{name}' has abnormally small scale: empirical std {emp_std:.6f} "
                      f"is {scale_ratio:.3f}x theoretical Kaiming std. Risk of vanishing signal.")

      # For square 2D matrices, compute spectral radius
      if data.dim() == 2 and data.shape[0] == data.shape[1] and data.shape[0] <= 1024:
        try:
          eigvals = torch.linalg.eigvals(data)
          rho = float(torch.max(torch.abs(eigvals)).item())
          spectral_radii[name] = round(rho, 4)

          if rho > 1.5:
            issues.append(
              f"Square matrix '{name}' has large spectral radius (rho = {rho:.2f} > 1.0). "
              "In recurrent or unscaled residual paths, this causes exponential explosion.")
        except Exception:
          pass

  if any("NaN" in iss or "zero variance" in iss for iss in issues):
    status = "failed"
  elif issues:
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "layer_stats": layer_stats,
    "spectral_radii": spectral_radii,
    "issues": issues,
  }
