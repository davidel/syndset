"""Tensor manipulation and loss resolution utilities for syndset."""

import torch
import torch.nn as nn


def extract_primary_tensor(output):
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

  if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
    return output.logits

  return None


_extract_primary_tensor = extract_primary_tensor


def resolve_loss(loss_fn=None, target=None):
  """Resolves the loss function, defaulting to CrossEntropyLoss or MSELoss based on target dtype.

  Args:
    loss_fn: Optional loss function provided by user.
    target: Optional target tensor used to infer appropriate loss function.

  Returns:
    A callable loss function.
  """
  if loss_fn is not None:
    return loss_fn

  if target is not None and hasattr(target, "dtype") and target.dtype in (torch.long, torch.int64):
    return nn.CrossEntropyLoss()

  return nn.MSELoss()
