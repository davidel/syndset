"""Numerical stability and mixed-precision sanity checks for neural networks."""

import torch


def check_numerical_stability(model, sample_input, check_half=True):
  """Checks numerical stability, NaN/Inf risk, and half-precision divergence.

  Modern training often uses FP16 or BF16. Certain architectures suffer from
  exponential activation scaling (e.g. Unscaled dot-products or missing layer norms)
  that easily overflows float16 (max ~65504) or causes underflow.

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
      "issues": [f"Forward pass failed with error: {str(e)}"],
    }

  # Extract primary tensor from output
  if isinstance(fp32_output, torch.Tensor):
    target_tensor = fp32_output
  elif (
    isinstance(fp32_output, (tuple, list))
    and len(fp32_output) > 0
    and isinstance(fp32_output[0], torch.Tensor)
  ):
    target_tensor = fp32_output[0]
  elif hasattr(fp32_output, "logits"):
    target_tensor = fp32_output.logits
  else:
    target_tensor = None

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
      "NaN detected in FP32 output! Check for division by zero or invalid log/sqrt operations."
    )
  if has_inf:
    issues.append(
      "Inf detected in FP32 output! Activations have overflowed standard 32-bit floating point."
    )

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
      "This will overflow IEEE 754 float16 (limit 65,504) during mixed-precision training."
    )

  # 2. Check bfloat16 stability if requested
  half_error = None
  if check_half:
    target_dtype = torch.bfloat16
    try:
      with torch.no_grad():
        if isinstance(sample_input, dict):
          half_input = {
            k: v.to(target_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for k, v in sample_input.items()
          }
        elif isinstance(sample_input, (tuple, list)):
          half_input = tuple(
            v.to(target_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v
            for v in sample_input
          )
        elif isinstance(sample_input, torch.Tensor) and sample_input.is_floating_point():
          half_input = sample_input.to(target_dtype)
        else:
          half_input = sample_input

        # Convert model temporarily to target_dtype
        model.to(target_dtype)
        if isinstance(half_input, dict):
          half_output = model(**half_input)
        elif isinstance(half_input, (tuple, list)):
          half_output = model(*half_input)
        else:
          half_output = model(half_input)

        # Restore model to float32
        model.to(torch.float32)

        if isinstance(half_output, torch.Tensor):
          h_tensor = half_output.float()
        elif (
          isinstance(half_output, (tuple, list))
          and len(half_output) > 0
          and isinstance(half_output[0], torch.Tensor)
        ):
          h_tensor = half_output[0].float()
        elif hasattr(half_output, "logits"):
          h_tensor = half_output.logits.float()
        else:
          h_tensor = None

        if h_tensor is not None:
          if torch.isnan(h_tensor).any() or torch.isinf(h_tensor).any():
            issues.append("BFloat16 forward pass produced NaNs or Infs despite FP32 succeeding.")
          else:
            diff_norm = torch.norm(h_tensor - target_tensor.float()).item()
            base_norm = torch.norm(target_tensor.float()).item()
            rel_error = diff_norm / (base_norm + 1e-7)
            half_error = float(rel_error)

            if half_error > 0.15:
              issues.append(
                f"Significant divergence under bfloat16 (norm diff: {half_error:.2%}). "
                "Consider keeping sensitive operations (e.g. Softmax) in FP32."
              )
    except Exception as e:
      model.to(torch.float32)
      issues.append(f"Half precision verification was skipped due to: {str(e)}")

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
