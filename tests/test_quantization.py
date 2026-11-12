"""Tests for activation kurtosis and quantization health diagnostics."""

import torch
import torch.nn as nn

from syndset.diagnostics.quantization import (
  check_quantization_health,
  compute_sample_kurtosis,
)


def test_compute_sample_kurtosis_normal():
  """Tests that standard normal samples yield kurtosis close to 3.0."""
  torch.manual_seed(42)
  samples = torch.randn(5000)
  kurt = compute_sample_kurtosis(samples)
  assert abs(kurt - 3.0) < 1.0


def test_check_quantization_health_normal():
  """Tests that a standard model has healthy activation kurtosis."""
  torch.manual_seed(42)
  model = nn.Sequential(
    nn.Linear(16, 32),
    nn.GELU(),
    nn.Linear(32, 8),
  )
  sample_input = torch.randn(16, 16)
  result = check_quantization_health(model, sample_input, kurtosis_threshold=50.0)

  assert result["status"] == "healthy"
  assert result["max_kurtosis"] < 50.0
  assert len(result["problematic_layers"]) == 0
