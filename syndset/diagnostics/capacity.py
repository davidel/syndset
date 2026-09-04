"""Capacity and single-batch memorization probes for neural networks."""

import copy

import torch
import torch.nn as nn
import torch.optim as optim


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


def check_overfit_capacity(
  model,
  inputs,
  targets,
  loss_fn=None,
  num_steps=100,
  learning_rate=1e-3,
  target_loss=1e-3,
  optimizer_cls=optim.AdamW,
):
  """Tests whether the model has enough parameter capacity to memorize a single batch.

  In deep learning engineering, a model that cannot drive loss on a small batch
  to near zero almost always has a structural defect (e.g. Broken skip connections,
  mismatched activation scales, or disconnected parameters).

  This test works on an isolated deep copy of the model weights so the original
  model parameters remain completely untouched.

  Args:
    model: The torch.nn.Module to evaluate.
    inputs: A torch.Tensor or tuple/dict of tensors for the model.
    targets: A torch.Tensor representing the expected target labels or values.
    loss_fn: Optional loss function. If None, automatically selects CrossEntropyLoss
      for integer labels or MSELoss for float targets.
    num_steps: Maximum number of optimization steps to attempt (default: 100).
    learning_rate: Learning rate for the optimizer (default: 1e-3).
    target_loss: The loss threshold below which the model is deemed to have
      successfully memorized the batch (default: 1e-3).
    optimizer_cls: Optimizer class to use (default: torch.optim.AdamW).

  Returns:
    A dictionary containing:
      - 'status': One of 'passed', 'slow', or 'failed'.
      - 'converged': Boolean indicating if target_loss was achieved.
      - 'steps_to_converge': The step index where target_loss was reached, or None.
      - 'initial_loss': Loss at step 0.
      - 'final_loss': Loss at the final optimization step.
      - 'loss_history': List of loss values recorded across steps.
      - 'issues': Human-readable notes or warnings.
  """
  # Select default loss function if omitted
  if loss_fn is None:
    if targets.dtype in (torch.long, torch.int64):
      loss_fn = nn.CrossEntropyLoss()
    else:
      loss_fn = nn.MSELoss()

  # Create an isolated clone of the model to avoid mutating caller's weights
  cloned_model = copy.deepcopy(model)
  cloned_model.train()

  optimizer = optimizer_cls(cloned_model.parameters(), lr=learning_rate)
  loss_history = []
  converged = False
  steps_to_converge = None

  for step in range(num_steps):
    optimizer.zero_grad()

    if isinstance(inputs, dict):
      outputs = cloned_model(**inputs)
    elif isinstance(inputs, (tuple, list)):
      outputs = cloned_model(*inputs)
    else:
      outputs = cloned_model(inputs)

    pred = _extract_primary_tensor(outputs)
    loss = loss_fn(pred if pred is not None else outputs, targets)
    loss_val = float(loss.item())
    loss_history.append(loss_val)

    if torch.isnan(loss) or torch.isinf(loss):
      return {
        "status": "failed",
        "converged": False,
        "steps_to_converge": None,
        "initial_loss": loss_history[0],
        "final_loss": loss_val,
        "loss_history": loss_history,
        "issues": [f"Loss exploded to {loss_val} at step {step}."],
      }

    if loss_val <= target_loss and not converged:
      converged = True
      steps_to_converge = step + 1
      break

    loss.backward()
    optimizer.step()

  initial_loss = loss_history[0]
  final_loss = loss_history[-1]
  issues = []

  if converged:
    if steps_to_converge <= num_steps // 3:
      status = "passed"
    else:
      status = "slow"
      issues.append(f"Memorized batch slowly (took {steps_to_converge}/{num_steps} steps). "
                    "Check learning rate or gradient scaling.")
  else:
    # Check if loss at least improved
    loss_ratio = final_loss / (initial_loss + 1e-10)
    if loss_ratio > 0.8:
      status = "failed"
      issues.append(
        f"Failed to memorize batch (final {final_loss:.4f} vs initial {initial_loss:.4f}). "
        "The model may lack capacity or have dead forward pathways.")
    else:
      status = "slow"
      issues.append(f"Loss decreased from {initial_loss:.4f} to {final_loss:.4f} but did not reach "
                    f"the target threshold of {target_loss} within {num_steps} steps.")

  return {
    "status": status,
    "converged": converged,
    "steps_to_converge": steps_to_converge,
    "initial_loss": initial_loss,
    "final_loss": final_loss,
    "loss_history": loss_history,
    "issues": issues,
  }
