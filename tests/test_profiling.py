"""Tests for efficiency and hardware profiling diagnostic."""

import torch
import torch.nn as nn

from syndset.diagnostics.profiling import (
  estimate_module_macs,
  profile_model_efficiency,
)


def test_estimate_module_macs():
  """Tests MAC calculation for linear and convolutional layers."""
  linear = nn.Linear(10, 20)
  in_t = torch.randn(2, 10)
  out_t = linear(in_t)
  macs = estimate_module_macs(linear, in_t, out_t)
  assert macs == 2 * 10 * 20


def test_profile_model_efficiency():
  """Tests model efficiency profiling."""
  model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
  )
  sample_input = torch.randn(4, 16)
  res = profile_model_efficiency(model, sample_input)

  assert res["status"] == "healthy"
  assert res["param_count"] == (16 * 32 + 32) + (32 * 4 + 4)
  assert res["trainable_params"] == res["param_count"]
  assert res["estimated_macs"] > 0
  assert res["param_memory_mb"] > 0.0
  assert res["activation_memory_mb"] > 0.0
