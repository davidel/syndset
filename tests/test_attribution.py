"""Unit tests for feature attribution, layer conductance, and causal faithfulness probes."""

import torch
import torch.nn as nn

from syndset.audit import audit, audit_attribution
from syndset.diagnostics.attribution import (
  check_attribution_faithfulness,
  check_context_attribution_decay,
  check_layer_conductance,
  compute_integrated_gradients,
)
from syndset.domains.llm import AssociativeRecallDataset, SelectiveCopyDataset
from syndset.domains.tabular import SparseXORDataset
from syndset.domains.vision import TextureVsShapeDataset


class ToyCausalModel(nn.Module):
  """Linear model that only connects to the first active_dim coordinates."""

  def __init__(self, total_dim=16, active_dim=3):
    super().__init__()
    self.fc = nn.Linear(total_dim, 2, bias=False)
    with torch.no_grad():
      self.fc.weight.zero_()
      # Non-zero weights only on active coordinates
      self.fc.weight[:, :active_dim] = 1.0

  def forward(self, x):
    return self.fc(x)


class ToyPathologicalDistractorModel(nn.Module):
  """Model that deliberately ignores active coordinates and only looks at noise."""

  def __init__(self, total_dim=16, active_dim=3):
    super().__init__()
    self.fc = nn.Linear(total_dim, 2, bias=False)
    with torch.no_grad():
      self.fc.weight.zero_()
      # Non-zero weights only on distractor coordinates
      self.fc.weight[:, active_dim:] = 2.0

  def forward(self, x):
    return self.fc(x)


class ToySeqModel(nn.Module):
  """Sequence model with embedding layer for discrete tokens."""

  def __init__(self, vocab_size=32, emb_dim=16, num_classes=32):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.proj = nn.Linear(emb_dim, num_classes)

  def forward(self, x):
    h = self.emb(x)
    return self.proj(h.mean(dim=1))


class ToyRecencyBiasedModel(nn.Module):
  """Sequence model that only looks at the final token."""

  def __init__(self, vocab_size=32, emb_dim=16, num_classes=32):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.proj = nn.Linear(emb_dim, num_classes)

  def forward(self, x):
    h = self.emb(x)
    # Ignores all tokens except the very last one
    return self.proj(h[:, -1, :])


class ModelWithDeadBranch(nn.Module):
  """3-layer network where the second layer is disconnected from output."""

  def __init__(self, in_dim=8, hidden_dim=8, out_dim=2):
    super().__init__()
    self.l1 = nn.Linear(in_dim, hidden_dim)
    self.dead_layer = nn.Linear(hidden_dim, hidden_dim)
    self.l3 = nn.Linear(hidden_dim, out_dim)

  def forward(self, x):
    h1 = torch.relu(self.l1(x))
    # dead_layer is calculated but disconnected from output
    _ = self.dead_layer(h1)
    return self.l3(h1)


def test_compute_integrated_gradients_completeness():
  """Verifies that Integrated Gradients satisfies the Completeness Axiom on float inputs."""
  torch.manual_seed(42)
  model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
  )
  inputs = torch.randn(4, 8)
  attrs, comp_err = compute_integrated_gradients(model, inputs, num_steps=25)

  assert attrs.shape == inputs.shape
  assert comp_err < 0.10


def test_compute_integrated_gradients_discrete_tokens():
  """Verifies path-integrated attribution for discrete sequence inputs."""
  torch.manual_seed(42)
  model = ToySeqModel()
  tokens = torch.randint(0, 32, (4, 10))
  attrs, comp_err = compute_integrated_gradients(model, tokens, num_steps=15)

  assert attrs.shape == tokens.shape
  assert comp_err < 0.20


def test_causal_mask_properties():
  """Tests causal_mask implementation on synthetic datasets."""
  xor_ds = SparseXORDataset(num_samples=20, total_dim=16, active_dim=4)
  mask_xor = xor_ds.causal_mask()
  assert mask_xor.shape == (16,)
  assert mask_xor[:4].all()
  assert not mask_xor[4:].any()

  ar_ds = AssociativeRecallDataset(num_samples=10, num_pairs=4, vocab_size=32)
  mask_ar = ar_ds.causal_mask(0)
  assert mask_ar.shape == (9,)
  assert mask_ar.sum() == 3  # key, value, query

  sc_ds = SelectiveCopyDataset(num_samples=10, num_signals=3, total_len=16)
  mask_sc = sc_ds.causal_mask(0)
  assert mask_sc.shape == (16,)
  assert mask_sc.sum() == 3

  tvs_ds = TextureVsShapeDataset(num_samples=10, img_size=16)
  mask_tvs = tvs_ds.causal_mask(0)
  assert mask_tvs.shape == (16, 16)


def test_check_attribution_faithfulness_healthy():
  """Tests that a model focused on causal features achieves high attribution SNR."""
  torch.manual_seed(42)
  dataset = SparseXORDataset(num_samples=30, total_dim=16, active_dim=3)
  model = ToyCausalModel(total_dim=16, active_dim=3)

  results = check_attribution_faithfulness(model, dataset, sample_indices=list(range(6)))
  assert results["status"] == "healthy"
  assert results["attribution_snr_db"] > 10.0
  assert results["causal_mass_ratio"] > 0.90
  assert results["topk_precision"] == 1.0


def test_check_attribution_faithfulness_pathological():
  """Tests that a model relying on distractor noise is flagged as failed."""
  torch.manual_seed(42)
  dataset = SparseXORDataset(num_samples=30, total_dim=16, active_dim=3)
  model = ToyPathologicalDistractorModel(total_dim=16, active_dim=3)

  results = check_attribution_faithfulness(model, dataset, sample_indices=list(range(6)))
  assert results["status"] == "failed"
  assert results["attribution_snr_db"] < 0.0
  assert any("places higher attribution on distractor noise" in iss for iss in results["issues"])


def test_check_layer_conductance_healthy():
  """Tests layer conductance on standard healthy MLP."""
  torch.manual_seed(42)
  model = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 4),
  )
  sample_input = torch.randn(4, 8)
  results = check_layer_conductance(model, sample_input, num_steps=10)

  assert results["status"] in ("healthy", "warning")
  assert len(results["layer_conductances"]) >= 2
  assert results["effective_depth_ratio"] > 0.4


def test_check_layer_conductance_dead_layer():
  """Tests detection of disconnected/dead layer in architecture."""
  torch.manual_seed(42)
  model = ModelWithDeadBranch(in_dim=8, hidden_dim=8, out_dim=2)
  sample_input = torch.randn(4, 8)
  results = check_layer_conductance(model, sample_input, num_steps=10)

  assert results["has_dead_layers"] is True
  assert results["status"] in ("warning", "failed")
  assert any("dead_layer" in iss for iss in results["issues"])


def test_check_context_attribution_decay():
  """Tests context attribution decay and recency bias detection."""
  torch.manual_seed(42)
  model = ToyRecencyBiasedModel(vocab_size=32, emb_dim=16, num_classes=32)
  tokens = torch.randint(0, 32, (4, 16))

  results = check_context_attribution_decay(model, tokens, num_steps=10)
  assert results["early_context_blindness"] is True
  assert results["status"] in ("warning", "failed")
  assert any("blindness" in iss for iss in results["issues"])


def test_audit_attribution_and_audit_conductance():
  """Tests audit_attribution and audit(run_conductance=True) reporting."""
  torch.manual_seed(42)
  dataset = SparseXORDataset(num_samples=20, total_dim=8, active_dim=2)
  model = ToyCausalModel(total_dim=8, active_dim=2)

  # Audit attribution
  report = audit_attribution(model, dataset, sample_indices=[0, 1])
  assert report.model_name == "ToyCausalModel"
  assert len(report.checks) == 1
  assert report.checks[0]["name"] == "Attribution Faithfulness"

  # Audit with conductance
  sample_input = torch.randn(4, 8)
  full_report = audit(model, sample_input, run_conductance=True)
  check_names = [c["name"] for c in full_report.checks]
  assert "Layer Conductance" in check_names
