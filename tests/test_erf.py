"""Tests for Effective Receptive Field (ERF) diagnostic."""

import torch
import torch.nn as nn

from syndset.diagnostics.erf import check_effective_receptive_field


def test_erf_probe_conv_network():
  """Tests ERF calculation on a convolutional network."""
  torch.manual_seed(42)
  model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 8, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(8, 2),
  )

  res = check_effective_receptive_field(model, input_shape=(1, 1, 16, 16))

  assert res["status"] in ("healthy", "warning")
  assert res["erf_radius_ratio"] > 0.0
  assert 0.0 <= res["checkerboard_ratio"] <= 1.0
