"""Diagnostics for evaluating gradient flow and parameter health."""

import math

import torch
import torch.nn as nn

from syndset.utils.stats import summarize_tensor


def _extract_primary_tensor(output):
  """Extracts the primary floating-point tensor from various model return structures.

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


def check_gradient_flow(model, sample_input, target=None, loss_fn=None):
  """Evaluates whether gradients backpropagate cleanly through all layers.

  Performs a single forward and backward step on a sample input to verify
  that every layer receives non-zero gradients, without suffering from
  vanishing gradients (norm near 0) or exploding gradients (norm excessively large).

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: A torch.Tensor or tuple/dict of tensors matching model input.
    target: Optional ground truth labels matching the loss function.
    loss_fn: Optional loss function callable. If None and target is provided,
      defaults to MSELoss or CrossEntropyLoss. If target is None, uses
      output.sum() to trace backward gradient pathways directly.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'issues': A list of human-readable diagnostic explanations.
      - 'layer_norms': A dict mapping parameter names to their gradient L2 norms.
      - 'global_norm': Total L2 norm across all parameters.
      - 'zero_grad_params': List of parameter names that received no gradient.
      - 'nan_grad_params': List of parameter names with NaN or Inf gradients.
      - 'vanishing_ratio': Ratio of smallest non-zero layer norm to largest layer norm.
  """
  model.train()
  model.zero_grad(set_to_none=True)

  # Handle input formats
  if isinstance(sample_input, dict):
    output = model(**sample_input)
  elif isinstance(sample_input, (tuple, list)):
    output = model(*sample_input)
  else:
    output = model(sample_input)

  pred = _extract_primary_tensor(output)

  # Determine and compute loss
  if loss_fn is not None:
    loss = loss_fn(pred if pred is not None else output, target)
  elif target is not None:
    if pred is None:
      raise ValueError("Could not extract a tensor from model output to compute loss.")
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
      err_msg = "Could not derive scalar loss from model output. Provide target or loss_fn."
      raise ValueError(err_msg)

  loss.backward()

  layer_norms = {}
  zero_grad_params = []
  nan_grad_params = []
  param_summaries = {}
  total_norm_sq = 0.0

  for name, param in model.named_parameters():
    if not param.requires_grad:
      continue
    summary = summarize_tensor(param.grad)
    param_summaries[name] = summary

    if summary["has_nan"] or summary["has_inf"]:
      nan_grad_params.append(name)
      continue

    grad_norm = summary["norm_l2"]
    layer_norms[name] = grad_norm
    total_norm_sq += grad_norm**2

    if grad_norm == 0.0:
      zero_grad_params.append(name)

  global_norm = math.sqrt(total_norm_sq)

  # Calculate vanishing ratio (min / max of non-zero gradient norms)
  non_zero_norms = [n for n in layer_norms.values() if n > 0.0]
  if non_zero_norms:
    min_norm = min(non_zero_norms)
    max_norm = max(non_zero_norms)
    vanishing_ratio = min_norm / (max_norm + 1e-12)
  else:
    vanishing_ratio = 0.0

  issues = []
  if nan_grad_params:
    issues.append(
      f"Exploding or invalid gradients detected: {len(nan_grad_params)} parameters "
      f"contain NaN or Inf values ({', '.join(nan_grad_params[:3])})."
    )

  if zero_grad_params:
    issues.append(
      f"Disconnected or dead parameters detected: {len(zero_grad_params)} trainable parameters "
      f"received zero gradient ({', '.join(zero_grad_params[:3])}). "
      "Check if these layers are detached or unused in forward pass."
    )

  if vanishing_ratio > 0.0 and vanishing_ratio < 1e-5:
    issues.append(
      f"Severe gradient vanishing detected: ratio between weakest and strongest layer gradient "
      f"is {vanishing_ratio:.2e}. Earlier layers may fail to learn."
    )

  if global_norm > 1e4:
    issues.append(
      f"Extremely large gradient norm ({global_norm:.2e}). "
      "This often indicates lack of normalization or unbounded activation growth."
    )

  if nan_grad_params or len(zero_grad_params) == len(list(model.parameters())):
    status = "failed"
  elif issues:
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "issues": issues,
    "layer_norms": layer_norms,
    "global_norm": global_norm,
    "zero_grad_params": zero_grad_params,
    "nan_grad_params": nan_grad_params,
    "vanishing_ratio": vanishing_ratio,
    "param_summaries": param_summaries,
  }


def check_gradient_snr(model, data_batches, loss_fn):
  """Estimates the Signal-to-Noise Ratio (SNR) of gradients across multiple batches.

  Calculates the mean gradient vector divided by standard deviation per parameter.
  A low SNR (< 0.1) suggests the batch size is too small or gradient updates are
  dominated by stochastic noise rather than a clear optimization direction.

  Args:
    model: The torch.nn.Module to evaluate.
    data_batches: A list or generator yielding (inputs, targets) pairs.
    loss_fn: Loss function callable(output, target).

  Returns:
    A dictionary mapping parameter names to their mean SNR value, along with
    a global aggregate SNR score.
  """
  model.train()
  grad_records = {}

  for inputs, targets in data_batches:
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    pred = _extract_primary_tensor(outputs)
    loss = loss_fn(pred if pred is not None else outputs, targets)
    loss.backward()

    for name, param in model.named_parameters():
      if param.requires_grad and param.grad is not None:
        if name not in grad_records:
          grad_records[name] = []
        grad_records[name].append(param.grad.detach().clone())

  param_snr = {}
  all_snrs = []

  for name, grads in grad_records.items():
    if len(grads) < 2:
      continue
    stacked = torch.stack(grads, dim=0).float()
    mean_grad = torch.mean(stacked, dim=0)
    std_grad = torch.std(stacked, dim=0)
    signal = torch.norm(mean_grad, 2).item()
    noise = torch.norm(std_grad, 2).item()
    snr = signal / (noise + 1e-10)
    param_snr[name] = float(snr)
    all_snrs.append(snr)

  avg_snr = sum(all_snrs) / max(len(all_snrs), 1)
  return {
    "average_snr": float(avg_snr),
    "parameter_snr": param_snr,
  }
