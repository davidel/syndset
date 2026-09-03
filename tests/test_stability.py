"""Tests for numerical stability probe."""

import torch
import torch.nn as nn

from syndset.diagnostics.stability import check_numerical_stability


def test_stability_healthy(toy_mlp):
  """Tests that a standard model reports healthy numerical stability."""
  inputs = torch.randn(8, 16)
  result = check_numerical_stability(toy_mlp, inputs)

  assert result["status"] == "healthy"
  assert not result["has_nan"]
  assert not result["has_inf"]
  assert result["max_abs_activation"] < 65000.0


def test_stability_nan_detection():
  """Tests that NaN occurrences in outputs are caught and flagged."""

  class NanModel(nn.Module):
    def forward(self, x):
      return x / 0.0

  model = NanModel()
  inputs = torch.randn(4, 4)
  result = check_numerical_stability(model, inputs)

  assert result["status"] == "failed"
  assert result["has_nan"] or result["has_inf"]
  assert len(result["issues"]) > 0
