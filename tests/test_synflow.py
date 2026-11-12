"""Tests for SynFlow data-free parameter saliency."""

import torch
import torch.nn as nn

from syndset.diagnostics.synflow import check_synflow


def test_synflow_healthy_model():
  """Tests that a standard feedforward model has positive SynFlow."""
  torch.manual_seed(42)
  model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
  )
  sample_input = torch.randn(4, 8)
  result = check_synflow(model, sample_input)

  assert result["status"] == "healthy"
  assert result["total_synflow"] > 0.0
  assert len(result["starved_layers"]) == 0


def test_synflow_disconnected_branch():
  """Tests that a dead parameter branch is identified as starved."""

  class DisconnectedModel(nn.Module):

    def __init__(self):
      super().__init__()
      self.active = nn.Linear(8, 2)
      self.dead = nn.Linear(8, 8)

    def forward(self, x):
      return self.active(x)

  torch.manual_seed(42)
  model = DisconnectedModel()
  sample_input = torch.randn(4, 8)
  result = check_synflow(model, sample_input)

  assert result["status"] in ("warning", "failed")
  assert any("dead" in name for name in result["starved_layers"])
