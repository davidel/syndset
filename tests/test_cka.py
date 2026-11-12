"""Tests for layer CKA similarity diagnostics."""

import torch
import torch.nn as nn

from syndset.diagnostics.cka import check_layer_cka, compute_linear_cka


def test_compute_linear_cka_identical():
  """Tests that CKA of identical feature matrices is 1.0."""
  torch.manual_seed(42)
  x = torch.randn(20, 16)
  sim = compute_linear_cka(x, x)
  assert abs(sim - 1.0) < 1e-4


def test_compute_linear_cka_orthogonal():
  """Tests that CKA of orthogonal features is near 0.0."""
  torch.manual_seed(42)
  x = torch.randn(100, 16)
  y = torch.randn(100, 16)
  sim = compute_linear_cka(x, y)
  assert 0.0 <= sim < 0.3


def test_check_layer_cka_on_model():
  """Tests CKA check on a multi-layer model."""
  torch.manual_seed(42)
  model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 4),
  )
  sample_input = torch.randn(8, 16)
  result = check_layer_cka(model, sample_input)

  assert result["status"] in ("healthy", "warning", "failed")
  assert len(result["layer_names"]) >= 3
  assert len(result["cka_matrix"]) == len(result["layer_names"])
  assert len(result["consecutive_similarities"]) >= 2
