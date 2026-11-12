"""Diagnostics for assessing activation kurtosis and Post-Training Quantization (PTQ) health."""

import torch

from syndset.diagnostics.representation import _flatten_activation_to_2d
from syndset.utils.hooks import ForwardActivationHook


def compute_sample_kurtosis(tensor, eps=1e-8):
  """Computes sample kurtosis of a 1D or flattened tensor.

  Theoretical background:
    Kurtosis = E[(X - mu)^4] / sigma^4.
    For a Gaussian distribution, kurtosis equals 3.0. Distributions with heavy tails
    (kurtosis >> 30) contain extreme activation outliers that collapse uniform INT8/FP8
    quantization bins, causing severe quantization error.

  Args:
    tensor: A torch.Tensor.
    eps: Small float to prevent division by zero.

  Returns:
    Float sample kurtosis value.
  """
  if tensor is None or tensor.numel() < 4:
    return 3.0

  x = tensor.float().view(-1)
  mean = x.mean()
  diff = x - mean
  variance = (diff**2).mean()

  if variance <= eps:
    return 3.0

  fourth_moment = (diff**4).mean()
  kurtosis = fourth_moment / (variance**2)
  return float(kurtosis.item())


def check_quantization_health(
  model,
  sample_input,
  target_types=None,
  kurtosis_threshold=30.0,
  outlier_sigma=6.0,
):
  """Evaluates activation distributions for INT8/FP8 quantization viability.

  Inspects intermediate activation distributions across layers. High kurtosis or
  a large ratio of extreme outliers (> 6 sigma) indicates that standard PTQ will
  experience precision collapse and severe degradation without costly custom schemes.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: Sample input tensor or tuple/dict matching forward signature.
    target_types: Optional module types to inspect. If None, inspects parameterized layers.
    kurtosis_threshold: Kurtosis threshold triggering a warning (default: 30.0).
    outlier_sigma: Number of standard deviations defining an outlier (default: 6.0).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'layer_kurtosis': Dict mapping layer name to computed kurtosis.
      - 'layer_outlier_ratios': Dict mapping layer name to outlier fraction.
      - 'problematic_layers': List of layer names exceeding thresholds.
      - 'max_kurtosis': Maximum kurtosis observed across all monitored layers.
      - 'issues': Human-readable recommendations for quantization readiness.
  """
  model.eval()

  with ForwardActivationHook(model, target_types=target_types) as hook:
    with torch.no_grad():
      if isinstance(sample_input, dict):
        _ = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        _ = model(*sample_input)
      else:
        _ = model(sample_input)

    activations = hook.activations

  layer_kurtosis = {}
  layer_outlier_ratios = {}
  problematic_layers = []
  issues = []
  max_kurtosis = 0.0

  for name, act in activations.items():
    flat_act = _flatten_activation_to_2d(act).float()
    if flat_act.numel() < 4:
      continue

    kurt = compute_sample_kurtosis(flat_act)
    layer_kurtosis[name] = kurt
    if kurt > max_kurtosis:
      max_kurtosis = kurt

    # Outlier ratio: elements > outlier_sigma * std from mean
    mean = flat_act.mean()
    std = flat_act.std()
    if std > 1e-8:
      outlier_mask = (flat_act - mean).abs() > (outlier_sigma * std)
      outlier_ratio = outlier_mask.float().mean().item()
    else:
      outlier_ratio = 0.0
    layer_outlier_ratios[name] = float(outlier_ratio)

    if kurt > kurtosis_threshold:
      problematic_layers.append(name)
      issues.append(
        f"Layer '{name}' has extreme activation kurtosis ({kurt:.1f} > {kurtosis_threshold}), "
        f"with {outlier_ratio:.2%} outliers > {outlier_sigma} sigma. Sensitive to PTQ quantization."
      )

  if len(problematic_layers) > len(layer_kurtosis) * 0.4:
    status = "failed"
  elif problematic_layers:
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "layer_kurtosis": layer_kurtosis,
    "layer_outlier_ratios": layer_outlier_ratios,
    "problematic_layers": problematic_layers,
    "max_kurtosis": float(max_kurtosis),
    "issues": issues,
  }
