"""Diagnostics for evaluating in-context copying and induction circuit plasticity."""

import torch
import torch.nn as nn

from syndset.diagnostics.curvature import _extract_primary_tensor
from syndset.domains.llm import InductionDataset


def check_induction_capability(
  model,
  vocab_size=64,
  seq_len=32,
  num_samples=16,
  num_steps=10,
  lr=1e-2,
  seed=42,
):
  """Evaluates whether an architecture can form in-context induction copying circuits.

  Theoretical background:
    Induction heads (Olsson et al., Anthropic 2022) are the primary mechanism enabling
    in-context learning in sequence models. In synthetic sequences [A, B, ..., A],
    the model must predict [B] by attending to prefix token [A] and copying the subsequent token.

    This probe generates procedural induction sequences and tracks loss decrease and
    copy accuracy over a micro-optimization budget on a single batch.

    Note on statistical power:
      The default parameters (num_samples=16, num_steps=10) serve as a rapid qualitative
      smoke test for the presence of induction learning mechanisms. For fine-grained comparative
      benchmarking across similar architectures, use higher sample sizes (e.g. num_samples=32 or 64)
      or average across multiple random seeds to mitigate sample variance.

  Args:
    model: The sequence torch.nn.Module to evaluate. Accepts long tensor (B, seq_len).
    vocab_size: Discrete vocabulary size (default: 64).
    seq_len: Sequence length (default: 32).
    num_samples: Number of probe samples (default: 16).
    num_steps: Micro-training optimization steps (default: 10).
    lr: Learning rate for fast probe optimizer (default: 1e-2).
    seed: Random seed for reproducible dataset generation (default: 42).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'initial_loss': Loss before micro-optimization steps.
      - 'final_loss': Loss after micro-optimization steps.
      - 'loss_reduction_ratio': Ratio of loss reduction.
      - 'initial_accuracy': Initial top-1 copy prediction accuracy.
      - 'final_accuracy': Final top-1 copy prediction accuracy.
      - 'converged': Boolean indicating whether induction was successfully learned.
      - 'issues': Human-readable recommendations.
  """
  # Auto-detect embedding vocabulary limit if model has an nn.Embedding layer
  for m in model.modules():
    if isinstance(m, nn.Embedding):
      vocab_size = min(vocab_size, m.num_embeddings)
      break

  dataset = InductionDataset(
    num_samples=num_samples,
    seq_len=seq_len,
    vocab_size=vocab_size,
    seed=seed,
  )
  inputs, targets = dataset[:num_samples]

  # Clone model parameters to restore state after probe
  orig_state = {k: v.clone() for k, v in model.state_dict().items()}
  model.train()

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  initial_loss = 0.0
  initial_acc = 0.0
  final_loss = 0.0
  final_acc = 0.0

  try:
    for step in range(num_steps):
      optimizer.zero_grad()
      outputs = model(inputs)
      logits = _extract_primary_tensor(outputs)

      if logits is None:
        return {
          "status": "failed",
          "initial_loss": float("inf"),
          "final_loss": float("inf"),
          "loss_reduction_ratio": 0.0,
          "initial_accuracy": 0.0,
          "final_accuracy": 0.0,
          "converged": False,
          "issues": ["Could not extract logits from model output."],
        }

      # If model outputs 3D logits (batch, seq_len, vocab_size), extract final token
      if logits.dim() == 3:
        logits = logits[:, -1, :]

      loss = loss_fn(logits, targets)
      preds = torch.argmax(logits, dim=-1)
      acc = (preds == targets).float().mean().item()

      if step == 0:
        initial_loss = float(loss.item())
        initial_acc = float(acc)

      loss.backward()
      optimizer.step()

      if step == num_steps - 1:
        final_loss = float(loss.item())
        final_acc = float(acc)

  finally:
    model.load_state_dict(orig_state)
    model.eval()

  reduction = max(0.0, (initial_loss - final_loss) / (initial_loss + 1e-8))
  converged = final_acc >= 0.8 or reduction >= 0.5

  issues = []
  if not converged:
    if reduction < 0.1:
      status = "failed"
      issues.append(
        f"Model failed induction probe (loss reduction: {reduction:.1%}, acc: {final_acc:.0%}). "
        f"Lacks mechanism to perform prefix in-context copying.")
    else:
      status = "warning"
      issues.append(
        f"Slow induction acquisition (acc: {final_acc:.0%}, loss reduction: {reduction:.1%}).")
  else:
    status = "healthy"

  return {
    "status": status,
    "initial_loss": initial_loss,
    "final_loss": final_loss,
    "loss_reduction_ratio": float(reduction),
    "initial_accuracy": initial_acc,
    "final_accuracy": final_acc,
    "converged": converged,
    "issues": issues,
  }
