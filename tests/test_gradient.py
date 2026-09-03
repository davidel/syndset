"""Tests for gradient flow probes and diagnostics."""

import torch
import torch.nn as nn

from syndset.diagnostics.gradient import check_gradient_flow, check_gradient_snr


def test_gradient_flow_healthy(toy_mlp):
  """Tests that gradient flow reports healthy on a normal MLP."""
  inputs = torch.randn(8, 16)
  targets = torch.randint(0, 3, (8,))
  result = check_gradient_flow(toy_mlp, inputs, target=targets)

  assert result["status"] == "healthy"
  assert len(result["zero_grad_params"]) == 0
  assert len(result["nan_grad_params"]) == 0
  assert result["global_norm"] > 0.0
  assert len(result["issues"]) == 0


def test_gradient_flow_disconnected_layer(disconnected_model):
  """Tests that unused parameters are successfully detected and flagged."""
  inputs = torch.randn(4, 8)
  result = check_gradient_flow(disconnected_model, inputs)

  assert result["status"] == "warning"
  assert len(result["zero_grad_params"]) > 0
  assert any("unused_layer" in name for name in result["zero_grad_params"])
  assert len(result["issues"]) > 0


def test_gradient_snr(toy_mlp):
  """Tests gradient SNR calculation over multiple batches."""
  loss_fn = nn.CrossEntropyLoss()
  batches = []
  for _ in range(3):
    x = torch.randn(8, 16)
    y = torch.randint(0, 3, (8,))
    batches.append((x, y))

  result = check_gradient_snr(toy_mlp, batches, loss_fn)
  assert "average_snr" in result
  assert result["average_snr"] >= 0.0
  assert len(result["parameter_snr"]) > 0
