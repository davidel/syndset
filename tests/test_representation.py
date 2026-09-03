"""Tests for representation diagnostics (effective rank and dead units)."""

import torch
import torch.nn as nn

from syndset.diagnostics.representation import check_dead_units, check_effective_rank


def test_effective_rank_mlp(toy_mlp):
  """Tests effective rank computation across MLP layers."""
  inputs = torch.randn(16, 16)
  result = check_effective_rank(toy_mlp, inputs)

  assert result["status"] == "healthy"
  assert len(result["layer_ranks"]) > 0
  for _name, erank in result["layer_ranks"].items():
    assert erank > 0.0


def test_effective_rank_convnet(toy_convnet):
  """Tests effective rank computation on 4D convolutional activations."""
  images = torch.randn(8, 1, 32, 32)
  result = check_effective_rank(toy_convnet, images)

  assert result["status"] == "healthy"
  assert len(result["layer_ranks"]) > 0


def test_dead_units_detection():
  """Tests that intentionally dead units (large negative bias + ReLU) are detected."""
  # Model with completely dead ReLU units due to strong negative bias
  dead_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
  with torch.no_grad():
    dead_model[0].bias.fill_(-1000.0)

  inputs = torch.randn(16, 8)
  result = check_dead_units(dead_model, inputs, dead_ratio_threshold=0.5)

  assert result["status"] == "warning"
  assert len(result["problematic_layers"]) > 0
