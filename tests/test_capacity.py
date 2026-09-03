"""Tests for capacity and single-batch overfit diagnostic."""

import torch

from syndset.diagnostics.capacity import check_overfit_capacity


def test_overfit_capacity_success(toy_mlp):
  """Tests that a small MLP can successfully overfit a small batch."""
  inputs = torch.randn(4, 16)
  targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

  # Record parameter clone to ensure original model is not modified
  orig_weights = toy_mlp.net[0].weight.clone()

  result = check_overfit_capacity(
    toy_mlp, inputs, targets, num_steps=60, learning_rate=0.05, target_loss=0.05
  )

  assert result["status"] in ("passed", "slow")
  assert result["final_loss"] < result["initial_loss"]
  assert torch.equal(toy_mlp.net[0].weight, orig_weights)


def test_overfit_capacity_failure():
  """Tests that an under-parameterized model failure is correctly detected."""
  # Model with 1 parameter trying to memorize conflicting targets
  single_param_layer = torch.nn.Linear(1, 1, bias=False)
  inputs = torch.tensor([[1.0], [1.0]])
  targets = torch.tensor([[5.0], [-5.0]])

  result = check_overfit_capacity(
    single_param_layer, inputs, targets, num_steps=20, target_loss=1e-4
  )

  assert result["status"] == "failed"
  assert not result["converged"]
  assert len(result["issues"]) > 0
