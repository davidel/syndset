"""Tests for in-context induction capability diagnostic."""

import torch
import torch.nn as nn

from syndset.diagnostics.induction import check_induction_capability


class MockSequenceModel(nn.Module):
  """Simple sequence model with embedding, GRU, and classifier head."""

  def __init__(self, vocab_size=64, dim=32):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, dim)
    self.gru = nn.GRU(dim, dim, batch_first=True)
    self.head = nn.Linear(dim, vocab_size)

  def forward(self, x):
    out, _ = self.gru(self.emb(x))
    return self.head(out[:, -1, :])


def test_induction_probe_execution():
  """Tests that the induction probe executes and restores original weights."""
  torch.manual_seed(42)
  model = MockSequenceModel(vocab_size=32, dim=16)

  orig_param = next(model.parameters()).clone()
  res = check_induction_capability(
    model,
    vocab_size=32,
    seq_len=16,
    num_samples=8,
    num_steps=3,
  )

  assert res["status"] in ("healthy", "warning", "failed")
  assert "initial_loss" in res
  assert "final_loss" in res
  assert "converged" in res

  # Verify weights were restored
  current_param = next(model.parameters())
  assert torch.allclose(orig_param, current_param)
