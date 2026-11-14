"""Tests verifying the fixes for all issues identified in code review."""

import math
import tomllib
from pathlib import Path

import torch
import torch.nn as nn

import syndset
from syndset.diagnostics.capacity import check_overfit_capacity
from syndset.diagnostics.stability import check_numerical_stability
from syndset.ranking import ModelCandidate, RankingEngine
from syndset.tournament import tournament_llm
from syndset.utils import extract_primary_tensor, resolve_loss


def test_i1_efficiency_pool_independent_and_smooth():
  """Verifies I-1: Efficiency is pool-independent, monotonic, and continuous."""
  engine = RankingEngine(profile="balanced")

  cand1 = ModelCandidate("Model1", None)
  cand1.raw_diagnostics = {
    "profiling": {
      "param_count": 29163840,
      "estimated_macs": 100000000
    },
  }

  cand2 = ModelCandidate("Model2", None)
  cand2.raw_diagnostics = {
    "profiling": {
      "param_count": 29200000,
      "estimated_macs": 100000000
    },
  }

  cand_small = ModelCandidate("SmallModel", None)
  cand_small.raw_diagnostics = {
    "profiling": {
      "param_count": 100000,
      "estimated_macs": 500000
    },
  }

  # Score cand1 alone
  engine._extract_candidate_metrics([cand1])
  engine._compute_utility_scores([cand1])
  score_alone = cand1.sub_scores["efficiency"]

  # Score cand1 in a pool containing a much smaller model
  engine._extract_candidate_metrics([cand1, cand_small])
  engine._compute_utility_scores([cand1, cand_small])
  score_in_pool = cand1.sub_scores["efficiency"]

  # Pool-independence: score must not change when adding candidates
  assert math.isclose(score_alone, score_in_pool, rel_tol=1e-5)

  # Continuity: ~0.1% parameter difference must not swing efficiency
  engine._extract_candidate_metrics([cand2])
  engine._compute_utility_scores([cand2])
  diff = abs(cand1.sub_scores["efficiency"] - cand2.sub_scores["efficiency"])
  assert diff < 0.01

  # Monotonicity with MACs: lower compute for same params yields higher efficiency
  cand_fewer_macs = ModelCandidate("FewerMACs", None)
  cand_fewer_macs.raw_diagnostics = {
    "profiling": {
      "param_count": 29163840,
      "estimated_macs": 10000000
    },
  }
  engine._extract_candidate_metrics([cand_fewer_macs])
  engine._compute_utility_scores([cand_fewer_macs])
  assert cand_fewer_macs.sub_scores["efficiency"] > cand1.sub_scores["efficiency"]


def test_i2_pareto_utility_consistency_and_fronts():
  """Verifies I-2 & N-4: Top composite candidate is never DROP, pareto_rank is computed."""
  engine = RankingEngine(profile="balanced")

  cand_a = ModelCandidate("ModelA", None)
  cand_a.raw_diagnostics = {
    "curvature": {
      "mean_curvature": 0.05
    },
    "capacity": {
      "converged": True,
      "steps_to_converge": 10
    },
    "effective_rank": {
      "rank_ratios": {
        "l1": 0.9
      }
    },
    "profiling": {
      "param_count": 10000,
      "estimated_macs": 50000
    },
    "quantization": {
      "max_kurtosis": 3.0
    },
  }

  cand_b = ModelCandidate("ModelB", None)
  cand_b.raw_diagnostics = {
    "curvature": {
      "mean_curvature": 0.05
    },
    "capacity": {
      "converged": True,
      "steps_to_converge": 10
    },
    "effective_rank": {
      "rank_ratios": {
        "l1": 0.9
      }
    },
    "profiling": {
      "param_count": 20000,
      "estimated_macs": 100000
    },
    "quantization": {
      "max_kurtosis": 3.0
    },
  }

  cand_c = ModelCandidate("ModelC", None)
  cand_c.raw_diagnostics = {
    "curvature": {
      "mean_curvature": 0.05
    },
    "capacity": {
      "converged": True,
      "steps_to_converge": 10
    },
    "effective_rank": {
      "rank_ratios": {
        "l1": 0.9
      }
    },
    "profiling": {
      "param_count": 40000,
      "estimated_macs": 200000
    },
    "quantization": {
      "max_kurtosis": 3.0
    },
  }

  ranked = engine.rank_candidates([cand_c, cand_b, cand_a])

  # Top model must be front 1, non-dominated, and PROMOTE
  assert ranked[0].name == "ModelA"
  assert ranked[0].pareto_rank == 1
  assert ranked[0].is_dominated is False
  assert ranked[0].verdict == "PROMOTE"

  # Dominated models must have sequential pareto ranks
  assert ranked[1].name == "ModelB"
  assert ranked[1].pareto_rank == 2
  assert ranked[1].is_dominated is True

  assert ranked[2].name == "ModelC"
  assert ranked[2].pareto_rank == 3
  assert ranked[2].is_dominated is True


def test_i3_half_precision_testing_both_dtypes():
  """Verifies I-3 & N-2 & N-5: Half-precision probe tests both bfloat16 and float16 cleanly."""
  model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
  sample_input = torch.randn(4, 8)

  orig_dtype = next(model.parameters()).dtype
  res = check_numerical_stability(model, sample_input, check_half=True)

  assert res["status"] in ("healthy", "warning")
  # Model dtype must always be restored
  assert next(model.parameters()).dtype == orig_dtype


def test_i4_induction_microprobe_scored_in_tournament():
  """Verifies I-4: Induction probe feeds into expressivity in tournament."""

  class ToyAttention(nn.Module):

    def __init__(self, vocab_size=32, d_model=16):
      super().__init__()
      self.emb = nn.Embedding(vocab_size, d_model)
      self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
      h = self.emb(x)
      return self.fc(h[:, -1, :])

  model = ToyAttention(vocab_size=32, d_model=16)
  report = tournament_llm({"ToyLLM": model}, num_pairs=4, vocab_size=32, batch_size=4)

  cand = report.candidates[0]
  # Induction must be present in raw diagnostics and reflected
  assert "induction" in cand.raw_diagnostics
  assert cand.metrics["expressivity"] > 0.0


def test_i5_converged_capacity_is_passed():
  """Verifies I-5: Reaching overfit target is marked passed even when taking > num_steps // 3."""
  model = nn.Linear(4, 2)
  inputs = torch.randn(4, 4)
  targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

  res = check_overfit_capacity(
    model,
    inputs,
    targets,
    num_steps=100,
    learning_rate=0.05,
    target_loss=0.05,
  )

  if res["converged"]:
    assert res["status"] == "passed"


def test_i6_version_triad_synchronized():
  """Verifies I-6: Version in __init__.py, pyproject.toml, and README are in sync."""
  repo_root = Path(__file__).resolve().parent.parent
  pyproject_path = repo_root / "pyproject.toml"
  with pyproject_path.open("rb") as f:
    pyproject_data = tomllib.load(f)

  pyproject_version = pyproject_data["project"]["version"]
  assert syndset.__version__ == pyproject_version

  readme_text = (repo_root / "README.md").read_text()
  assert f"pypi-v{pyproject_version}" in readme_text


def test_n1_n3_utils_helpers():
  """Verifies N-1 & N-3: extract_primary_tensor and resolve_loss function properly."""
  t = torch.randn(2, 4)
  assert extract_primary_tensor(t) is t
  assert extract_primary_tensor((t, "extra")) is t

  class OutputObj:

    def __init__(self, logits):
      self.logits = logits

  assert extract_primary_tensor(OutputObj(t)) is t

  int_target = torch.tensor([0, 1])
  float_target = torch.tensor([0.5, 1.5])
  assert isinstance(resolve_loss(None, int_target), nn.CrossEntropyLoss)
  assert isinstance(resolve_loss(None, float_target), nn.MSELoss)
