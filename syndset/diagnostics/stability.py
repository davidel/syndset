"""Numerical stability and mixed-precision sanity checks for neural networks."""

import torch

from syndset.utils import extract_primary_tensor


def _cast_inputs(sample_input, target_dtype):
  """Recursively casts floating point tensors in sample_input to target_dtype."""
  if isinstance(sample_input, dict):
    return {
      k: v.to(target_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
      for k, v in sample_input.items()
    }

  if isinstance(sample_input, (tuple, list)):
    return tuple(
      v.to(target_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
      for v in sample_input)

  if isinstance(sample_input, torch.Tensor) and sample_input.is_floating_point():
    return sample_input.to(target_dtype)

  return sample_input


def check_numerical_stability(model, sample_input, check_half=True):
  """Checks numerical stability, NaN/Inf risk, and half-precision divergence.

  Theoretical background:
    In IEEE 754 floating-point standards:
      - FP32: 8 exponent bits, 23 mantissa bits, range ~10^38, epsilon ~1.19e-7.
      - BF16: 8 exponent bits, 7 mantissa bits, range ~10^38, epsilon ~7.81e-3.
      - FP16: 5 exponent bits, 10 mantissa bits, MAX VALUE = 65,504, epsilon ~9.77e-4.

    In attention layers, unscaled dot products Var(q^T k) = d_k produce logits
    reaching +/-3*sqrt(d_k) (e.g. +/-34 for d_k = 128). In FP16, exp(34) ~ 5.8e14,
    exceeding 65,504 and causing immediate overflow to +inf and NaN gradients.

    This probe compares FP32 outputs against half-precision (bfloat16 and float16)
    using the relative Frobenius norm:
      rel_err = ||Y_half - Y_fp32||_F / (||Y_fp32||_F + 1e-7)
    If rel_err > 0.15, internal variance reductions or activations suffer from
    catastrophic cancellation and must remain in FP32.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: A torch.Tensor or tuple/dict of tensors in float32.
    check_half: Whether to compare float32 output with half-precision (float16/bfloat16).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'has_nan': True if NaN was produced.
      - 'has_inf': True if Inf was produced.
      - 'max_abs_activation': Maximum absolute activation value observed in output.
      - 'half_precision_relative_error': Relative Frobenius norm difference in half precision.
      - 'issues': Human-readable notes and warnings.
  """
  model.eval()
  issues = []

  # 1. Base FP32 forward pass
  try:
    with torch.no_grad():
      if isinstance(sample_input, dict):
        fp32_output = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        fp32_output = model(*sample_input)
      else:
        fp32_output = model(sample_input)
  except Exception as e:
    return {
      "status": "failed",
      "has_nan": False,
      "has_inf": False,
      "max_abs_activation": 0.0,
      "half_precision_relative_error": None,
      "issues": [f"Forward pass failed with {type(e).__name__}: {str(e)}"],
    }

  target_tensor = extract_primary_tensor(fp32_output)

  if target_tensor is None:
    return {
      "status": "warning",
      "has_nan": False,
      "has_inf": False,
      "max_abs_activation": 0.0,
      "half_precision_relative_error": None,
      "issues": ["Could not parse a torch.Tensor from model output to inspect stability."],
    }

  has_nan = bool(torch.isnan(target_tensor).any().item())
  has_inf = bool(torch.isinf(target_tensor).any().item())

  if has_nan:
    issues.append(
      "NaN detected in FP32 output! Check for division by zero or invalid log/sqrt operations.")
  if has_inf:
    issues.append(
      "Inf detected in FP32 output! Activations have overflowed standard 32-bit floating point.")

  if has_nan or has_inf:
    return {
      "status": "failed",
      "has_nan": has_nan,
      "has_inf": has_inf,
      "max_abs_activation": float("inf"),
      "half_precision_relative_error": None,
      "issues": issues,
    }

  max_abs_val = float(torch.max(torch.abs(target_tensor)).item())
  if max_abs_val > 65000.0:
    issues.append(
      f"High maximum activation magnitude ({max_abs_val:.1f}). "
      "This will overflow IEEE 754 float16 (limit 65,504) during mixed-precision training.")

  # 2. Check half-precision stability (bfloat16 and float16)
  half_error = None
  if check_half:
    # Test bfloat16 first (wide dynamic range) then float16 (standard 16-bit)
    dtypes_to_test = [("BFloat16", torch.bfloat16), ("Float16", torch.float16)]
    errors = []

    for dtype_name, target_dtype in dtypes_to_test:
      try:
        with torch.no_grad():
          half_input = _cast_inputs(sample_input, target_dtype)
          try:
            model.to(target_dtype)
            if isinstance(half_input, dict):
              half_output = model(**half_input)
            elif isinstance(half_input, (tuple, list)):
              half_output = model(*half_input)
            else:
              half_output = model(half_input)
          finally:
            model.to(torch.float32)

          h_tensor = extract_primary_tensor(half_output)
          if h_tensor is not None:
            h_tensor = h_tensor.float()
            if torch.isnan(h_tensor).any() or torch.isinf(h_tensor).any():
              issues.append(
                f"{dtype_name} forward pass produced NaNs or Infs despite FP32 succeeding.")
            else:
              diff_norm = torch.norm(h_tensor - target_tensor.float()).item()
              base_norm = torch.norm(target_tensor.float()).item()
              rel_error = diff_norm / (base_norm + 1e-7)
              errors.append(float(rel_error))

              if rel_error > 0.15:
                issues.append(
                  f"Significant divergence under {dtype_name} (norm diff: {rel_error:.2%}). "
                  "Consider keeping sensitive operations (e.g. Softmax) in FP32.")
      except Exception as e:
        model.to(torch.float32)
        issues.append(f"{dtype_name} verification was skipped due to {type(e).__name__}: {str(e)}")

    if errors:
      half_error = errors[0]

  if has_nan or has_inf:
    status = "failed"
  elif issues:
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "has_nan": has_nan,
    "has_inf": has_inf,
    "max_abs_activation": max_abs_val,
    "half_precision_relative_error": half_error,
    "issues": issues,
  }
