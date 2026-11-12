"""Tests for empirical NTK conditioning diagnostic."""

import torch
import torch.nn as nn

from syndset.diagnostics.ntk import check_ntk_conditioning


def test_ntk_conditioning_linear():
  """Tests empirical NTK computation on a simple linear regression model."""
  torch.manual_seed(42)
  model = nn.Linear(8, 2)
  sample_input = torch.randn(4, 8)
  result = check_ntk_conditioning(model, sample_input, max_samples=4)

  assert result["status"] in ("healthy", "warning")
  assert result["condition_number"] >= 1.0
  assert len(result["eigenvalues"]) == 4
  assert result["trace"] > 0.0


def test_ntk_degenerate_input():
  """Tests handling when fewer than 2 samples are provided."""
  model = nn.Linear(4, 2)
  sample_input = torch.randn(1, 4)
  result = check_ntk_conditioning(model, sample_input)
  assert result["status"] == "healthy"
