"""Tests for advanced diagnostic probes: initialization, curvature, and permutation."""

import torch
import torch.nn as nn

from syndset.diagnostics.curvature import check_curvature_sharpness
from syndset.diagnostics.initialization import check_initialization_scale
from syndset.diagnostics.permutation import check_permutation_equivariance


class SimpleEquivariantModel(nn.Module):
  """Linear layer applied independently to each token/element across sequence dimension."""

  def __init__(self, in_dim=8, out_dim=8):
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    return self.linear(x)


class SimpleInvariantModel(nn.Module):
  """Sum pooling across sequence dimension, producing permutation invariant output."""

  def __init__(self, in_dim=8, out_dim=4):
    super().__init__()
    self.linear = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    return self.linear(x).sum(dim=1)


def test_initialization_scale_healthy():
  """Tests initialization scale verification on standard PyTorch initialized model."""
  model = nn.Sequential(
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
  )
  results = check_initialization_scale(model)
  assert results["status"] in ("healthy", "warning")
  assert len(results["layer_stats"]) >= 2
  assert "0.weight" in results["layer_stats"]


def test_initialization_scale_unstable():
  """Tests detection of explosive weights and large spectral radius."""
  model = nn.Linear(32, 32, bias=False)
  # Artificially set explosive weights
  with torch.no_grad():
    model.weight.fill_(10.0)

  results = check_initialization_scale(model, scale_tolerance_ratio=5.0)
  assert results["status"] in ("warning", "failed")
  has_scale_issue = any(
    "abnormally large scale" in iss or "spectral radius" in iss for iss in results["issues"])
  assert has_scale_issue


def test_curvature_sharpness_calculation():
  """Tests Hutchinson Hessian trace calculation on a toy model."""
  model = nn.Sequential(
    nn.Linear(8, 16),
    nn.Tanh(),
    nn.Linear(16, 2),
  )
  inputs = torch.randn(4, 8)
  targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

  results = check_curvature_sharpness(model, inputs, target=targets, num_samples=3)
  assert results["status"] in ("healthy", "warning")
  assert isinstance(results["hessian_trace"], float)
  assert isinstance(results["mean_curvature"], float)


def test_permutation_equivariance_success():
  """Tests permutation equivariance probe on an element-wise mapping."""
  model = SimpleEquivariantModel(in_dim=8, out_dim=8)
  # Input shape: (batch_size, num_elements, feature_dim)
  sample_input = torch.randn(2, 6, 8)
  results = check_permutation_equivariance(model, sample_input, perm_dim=1, check_invariance=False)
  assert results["is_satisfied"] is True
  assert results["relative_difference"] < 1e-4
  assert results["status"] == "healthy"


def test_permutation_invariance_success():
  """Tests permutation invariance probe on a sum-pooling network."""
  model = SimpleInvariantModel(in_dim=8, out_dim=4)
  sample_input = torch.randn(2, 6, 8)
  results = check_permutation_equivariance(model, sample_input, perm_dim=1, check_invariance=True)
  assert results["is_satisfied"] is True
  assert results["relative_difference"] < 1e-4
  assert results["status"] == "healthy"


def test_permutation_equivariance_failure():
  """Tests detection of broken equivariance when positional biases are added."""

  class NonEquivariantModel(nn.Module):

    def __init__(self):
      super().__init__()
      # Add static positional weights
      self.pos = nn.Parameter(torch.randn(1, 6, 8))

    def forward(self, x):
      return x + self.pos

  model = NonEquivariantModel()
  sample_input = torch.randn(2, 6, 8)
  results = check_permutation_equivariance(model, sample_input, perm_dim=1, check_invariance=False)
  assert results["is_satisfied"] is False
  assert results["status"] == "warning"
