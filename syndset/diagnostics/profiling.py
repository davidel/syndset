"""Diagnostics for systems profiling: parameters, FLOPs/MACs, and activation memory."""

import torch
import torch.nn as nn


def estimate_module_macs(module, input_tensor, output_tensor):
  """Estimates multiply-accumulate operations (MACs) for common neural layers.

  Args:
    module: The PyTorch nn.Module.
    input_tensor: Forward input tensor.
    output_tensor: Forward output tensor.

  Returns:
    Integer estimated MAC count.
  """
  macs = 0
  if isinstance(module, nn.Linear):
    # in_features * out_features * batch_elements
    batch_numel = input_tensor.numel() // module.in_features
    macs = module.in_features * module.out_features * batch_numel
  elif isinstance(module, nn.Conv2d):
    # out_channels * in_channels / groups * k_h * k_w * out_h * out_w * batch
    if output_tensor.dim() == 4:
      batch, _, out_h, out_w = output_tensor.shape
      k_h, k_w = module.kernel_size
      in_c = module.in_channels // module.groups
      macs = batch * module.out_channels * in_c * k_h * k_w * out_h * out_w
  elif isinstance(
      module,
    (nn.LayerNorm, nn.BatchNorm2d, nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.Module),
  ):
    macs = input_tensor.numel()
  return macs


def profile_model_efficiency(model, sample_input):
  """Profiles parameter efficiency, estimated MACs, and activation memory footprint.

  Args:
    model: The torch.nn.Module to inspect.
    sample_input: Sample input matching forward signature.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'param_count': Total parameter count.
      - 'trainable_params': Number of trainable parameters.
      - 'param_memory_mb': Estimated memory used by parameters in MB.
      - 'estimated_macs': Total estimated Multiply-Accumulate operations.
      - 'activation_memory_mb': Memory required to hold forward activations in MB.
      - 'issues': Informational notes on hardware efficiency.
  """
  model.eval()

  total_params = 0
  trainable_params = 0
  param_bytes = 0

  for p in model.parameters():
    numel = p.numel()
    total_params += numel
    if p.requires_grad:
      trainable_params += numel
    param_bytes += numel * p.element_size()

  param_memory_mb = param_bytes / (1024 * 1024)

  total_macs = 0
  activation_bytes = 0

  def _profile_hook(module, inp, out):
    nonlocal total_macs, activation_bytes
    if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
      in_t = inp[0]
    else:
      in_t = None

    if isinstance(out, torch.Tensor):
      out_t = out
      activation_bytes += out.numel() * out.element_size()
    elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
      out_t = out[0]
      for item in out:
        if isinstance(item, torch.Tensor):
          activation_bytes += item.numel() * item.element_size()
    else:
      out_t = None

    if in_t is not None and out_t is not None:
      total_macs += estimate_module_macs(module, in_t, out_t)

  handles = []
  for module in model.modules():
    if module != model and len(list(module.children())) == 0:
      handle = module.register_forward_hook(_profile_hook)
      handles.append(handle)

  try:
    with torch.no_grad():
      if isinstance(sample_input, dict):
        _ = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        _ = model(*sample_input)
      else:
        _ = model(sample_input)
  finally:
    for h in handles:
      h.remove()

  act_memory_mb = activation_bytes / (1024 * 1024)

  issues = []
  if total_params == 0:
    status = "failed"
    issues.append("Model has 0 parameters.")
  elif act_memory_mb > param_memory_mb * 20.0 and act_memory_mb > 50.0:
    status = "warning"
    issues.append(
      f"High activation memory footprint ({act_memory_mb:.1f} MB vs "
      f"{param_memory_mb:.1f} MB weights). May require gradient checkpointing during training.")
  else:
    status = "healthy"

  return {
    "status": status,
    "param_count": total_params,
    "trainable_params": trainable_params,
    "param_memory_mb": float(param_memory_mb),
    "estimated_macs": int(total_macs),
    "activation_memory_mb": float(act_memory_mb),
    "issues": issues,
  }
