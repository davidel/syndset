"""Tests verifying the fixes for all issues identified in SD_ISSUES.md."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from syndset.audit import audit_llm
from syndset.diagnostics.cka import check_layer_cka
from syndset.diagnostics.curvature import check_curvature_sharpness
from syndset.diagnostics.gradient import check_gradient_snr
from syndset.diagnostics.ntk import check_ntk_conditioning
from syndset.diagnostics.synflow import check_synflow
from syndset.ranking import ModelCandidate, RankingEngine
from syndset.tournament import TournamentReport, tournament_llm


class CausalLM3D(nn.Module):
  """Standard causal LM that returns 3D logits (batch, seq_len, vocab_size)."""

  def __init__(self, vocab_size=64, d_model=32):
    super().__init__()
    self.wte = nn.Embedding(vocab_size, d_model)
    self.wpe = nn.Embedding(64, d_model)
    self.head = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    pos = torch.arange(x.shape[1], device=x.device)
    h = self.wte(x) + self.wpe(pos)
    # Returns 3D tensor without manual sequence slicing
    return self.head(h)


class SDPAAttentionModel(nn.Module):
  """Model using PyTorch F.scaled_dot_product_attention on CPU."""

  def __init__(self, d_model=32, num_heads=2):
    super().__init__()
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)
    self.head = nn.Linear(d_model, 2)
    self.num_heads = num_heads
    self.d_head = d_model // num_heads

  def forward(self, x):
    b, s, d = x.shape
    q = self.q_proj(x).view(b, s, self.num_heads, self.d_head).transpose(1, 2)
    k = self.k_proj(x).view(b, s, self.num_heads, self.d_head).transpose(1, 2)
    v = self.v_proj(x).view(b, s, self.num_heads, self.d_head).transpose(1, 2)
    attn = F.scaled_dot_product_attention(q, k, v)
    attn = attn.transpose(1, 2).contiguous().view(b, s, d)
    out = self.out_proj(attn)
    return self.head(out[:, -1, :])


def test_a1_causal_lm_3d_output_in_audit_and_tournament():
  """Tests A1: Models returning 3D logits (B, T, V) work in audit_llm and tournament_llm."""
  model = CausalLM3D(vocab_size=64, d_model=32)

  # Should not raise RuntimeError: Expected target size [8, 256], got [8]
  report = audit_llm(model, num_pairs=4, vocab_size=64, batch_size=4)
  assert report.is_healthy()

  tourney_report = tournament_llm({"CausalLM": model}, num_pairs=4, vocab_size=64, batch_size=4)
  assert len(tourney_report.candidates) == 1
  assert tourney_report.candidates[0].composite_score > 0.0


def test_a2_sdpa_curvature_on_cpu():
  """Tests A2: Curvature sharpness executes cleanly with SDPA on CPU without derivative failure."""
  model = SDPAAttentionModel(d_model=32, num_heads=2)
  sample_input = torch.randn(4, 8, 32)
  target = torch.randint(0, 2, (4,))

  res = check_curvature_sharpness(model, sample_input, target=target)
  assert res["status"] in ("healthy", "warning")
  assert isinstance(res["mean_curvature"], float)


def test_b1_cka_embedding_filtering():
  """Tests B1: CKA excludes embedding layers by default, avoiding false disconnects."""
  model = CausalLM3D(vocab_size=64, d_model=32)
  sample_input = torch.randint(0, 64, (4, 16))

  res = check_layer_cka(model, sample_input)
  # Embedding layers wte and wpe must not be hooked and flagged as disconnected
  for pair in res["disconnected_layers"]:
    assert "wte" not in pair and "wpe" not in pair
  assert res["status"] == "healthy"


def test_b2_synflow_with_zero_initialized_biases():
  """Tests B2: Zero-initialized affine biases do not cause >30% starvation failure."""
  model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
  )
  # Explicitly zero all biases
  for m in model.modules():
    if isinstance(m, nn.Linear):
      nn.init.zeros_(m.bias)

  sample_input = torch.randn(4, 16)
  res = check_synflow(model, sample_input)
  assert res["status"] == "healthy"
  assert len(res["starved_layers"]) == 0


def test_c1_and_c2_tournament_report_and_continuous_cap_score():
  """Tests C1 & C2: TournamentReport renders sub-scores and continuous cap_score."""
  engine = RankingEngine(profile="balanced")

  cand_a = ModelCandidate("ModelA", nn.Linear(8, 2))
  cand_a.raw_diagnostics = {
    "capacity": {"converged": False, "initial_loss": 2.5, "final_loss": 0.5},
    "effective_rank": {"rank_ratios": {"fc": 0.6}},
    "profiling": {"param_count": 1000},
  }

  cand_b = ModelCandidate("ModelB", nn.Linear(8, 2))
  cand_b.raw_diagnostics = {
    "capacity": {"converged": False, "initial_loss": 2.5, "final_loss": 2.4},
    "effective_rank": {"rank_ratios": {"fc": 0.6}},
    "profiling": {"param_count": 1000},
  }

  ranked = engine.rank_candidates([cand_a, cand_b])
  # ModelA reduced loss significantly more, so its expressivity / score must be higher
  assert ranked[0].name == "ModelA"
  assert ranked[0].composite_score > ranked[1].composite_score

  report = TournamentReport(ranked, domain="llm", profile="balanced")
  summary_text = report.summary()
  assert "Train" in summary_text
  assert "Expr" in summary_text
  assert "Eff" in summary_text
  assert "Quant" in summary_text


def test_d1_gradient_snr_signature_flexibility():
  """Tests D1: check_gradient_snr accepts both iterable of pairs and (inputs, targets) tensors."""
  model = nn.Linear(8, 2)
  inputs = torch.randn(8, 8)
  targets = torch.randint(0, 2, (8,))

  # Standard (model, inputs, targets) call
  res1 = check_gradient_snr(model, inputs, targets)
  assert "average_snr" in res1

  # Iterable of batches call
  batches = [(inputs[:4], targets[:4]), (inputs[4:], targets[4:])]
  res2 = check_gradient_snr(model, batches, nn.CrossEntropyLoss())
  assert "average_snr" in res2


def test_d2_ntk_conditioning_signature_uniformity():
  """Tests D2: check_ntk_conditioning accepts sample_target without error."""
  model = nn.Linear(8, 2)
  sample_input = torch.randn(4, 8)
  sample_target = torch.randint(0, 2, (4,))

  # Calling with sample_target should not collide with max_samples
  res = check_ntk_conditioning(model, sample_input, sample_target)
  assert res["status"] in ("healthy", "warning")
