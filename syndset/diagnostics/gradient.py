"""Diagnostics for evaluating gradient flow and parameter health."""

import math

import torch

from syndset.utils import extract_primary_tensor, resolve_loss
from syndset.utils.stats import summarize_tensor

_extract_primary_tensor = extract_primary_tensor


def check_gradient_flow(model, sample_input, target=None, loss_fn=None):
  """Evaluates whether gradients backpropagate cleanly through all layers.

  Theoretical background:
    For an L-layer network h_l = f_l(h_{l-1}, theta_l), the chain rule gives:
      dL / dh_0 = (dL / dh_L) * prod_{l=1}^L J_l
    where J_l = dh_l / dh_{l-1} is the layer Jacobian matrix.
    If the spectral norms ||J_l||_2 < 1 - epsilon across layers, gradients vanish
    exponentially: ||dL / dh_0|| <= O((1 - epsilon)^L) -> 0.
    Conversely, if ||J_l||_2 > 1 + epsilon, gradients explode: O((1 + epsilon)^L) -> inf.

  This probe executes a single forward/backward pass and calculates:
    - Layer-wise gradient norms: ||grad_{theta_l} L||_2
    - Vanishing ratio: R_vanish = min_l ||grad_l|| / (max_l ||grad_l|| + 1e-12)
    - Zero-gradient parameters: indicates disconnected branches or detached tensors.

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

  # Normalize 3D sequence predictions (batch, seq_len, vocab) to 2D next-token logits (batch, vocab)
  # when evaluating against 1D scalar sequence targets.
  if pred is not None and target is not None and pred.dim() == 3 and target.dim() == 1:
    pred = pred[:, -1, :]

  # Determine and compute loss
  if target is not None:
    if pred is None:
      raise ValueError("Could not extract a tensor from model output to compute loss.")
    loss = resolve_loss(loss_fn, target)(pred, target)
  elif loss_fn is not None:
    loss = loss_fn(pred if pred is not None else output)
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
    issues.append(f"Exploding or invalid gradients detected: {len(nan_grad_params)} parameters "
                  f"contain NaN or Inf values ({', '.join(nan_grad_params[:3])}).")

  if zero_grad_params:
    issues.append(
      f"Disconnected or dead parameters detected: {len(zero_grad_params)} trainable parameters "
      f"received zero gradient ({', '.join(zero_grad_params[:3])}). "
      "Check if these layers are detached or unused in forward pass.")

  if vanishing_ratio > 0.0 and vanishing_ratio < 1e-5:
    issues.append(
      f"Severe gradient vanishing detected: ratio between weakest and strongest layer gradient "
      f"is {vanishing_ratio:.2e}. Earlier layers may fail to learn.")

  if global_norm > 1e4:
    issues.append(f"Extremely large gradient norm ({global_norm:.2e}). "
                  "This often indicates lack of normalization or unbounded activation growth.")

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


def check_gradient_snr(model, data_batches, target_or_loss_fn=None, loss_fn=None):
  """Estimates the Signal-to-Noise Ratio (SNR) of gradients across multiple batches.

  Theoretical formulation:
    Over K minibatches with gradient vectors g_k = grad_theta L(B_k):
      Sample mean vector: g_bar = (1 / K) * sum_k g_k
      Coordinate variance: s_i^2 = (1 / (K - 1)) * sum_k (g_{k,i} - g_bar_i)^2
      Signal-to-Noise Ratio: SNR(theta) = ||g_bar||_2 / sqrt(sum_i s_i^2)

  Interpretation:
    - SNR > 1.0: Consistent descent signal across minibatches.
    - SNR < 0.1: Noise-dominated updates, suggesting batch size is too small or
      the loss surface has high variance.

  Args:
    model: The torch.nn.Module to evaluate.
    data_batches: Either a list/generator yielding (inputs, targets) pairs,
      OR a single batch tensor matching model inputs.
    target_or_loss_fn: Either loss_fn callable (when data_batches is a list of pairs),
      OR target tensor matching standard probe signatures (model, inputs, targets, loss_fn).
    loss_fn: Optional loss function callable(output, target). If None, defaults
      to CrossEntropyLoss or MSELoss based on target dtype.

  Returns:
    A dictionary mapping parameter names to their mean SNR value, along with
    a global aggregate SNR score.
  """
  model.train()
  grad_records = {}

  # Polymorphic argument resolution: handle both (model, batches, loss_fn)
  # and (model, inputs, targets, loss_fn) call patterns.
  if callable(target_or_loss_fn) and loss_fn is None:
    effective_loss_fn = target_or_loss_fn
    batches = data_batches
  elif target_or_loss_fn is not None and not callable(target_or_loss_fn):
    effective_loss_fn = loss_fn
    inputs = data_batches
    targets = target_or_loss_fn
    if isinstance(inputs, torch.Tensor) and inputs.shape[0] >= 2:
      mid = inputs.shape[0] // 2
      batches = [(inputs[:mid], targets[:mid]), (inputs[mid:], targets[mid:])]
    else:
      batches = [(inputs, targets)]
  else:
    effective_loss_fn = loss_fn
    batches = data_batches

  for inputs, targets in batches:
    model.zero_grad(set_to_none=True)
    outputs = model(inputs)
    pred = _extract_primary_tensor(outputs)

    # Normalize 3D sequence predictions (batch, seq_len, vocab) to 2D next-token logits
    if pred is not None and targets is not None and pred.dim() == 3 and targets.dim() == 1:
      pred = pred[:, -1, :]

    if targets is not None:
      effective_target = targets
      criterion = resolve_loss(effective_loss_fn, effective_target)
      loss = criterion(pred if pred is not None else outputs, targets)
    elif effective_loss_fn is not None:
      loss = effective_loss_fn(pred if pred is not None else outputs)
    else:
      loss = (pred if pred is not None else outputs).sum()

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
