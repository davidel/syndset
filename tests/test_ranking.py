"""Tests for multi-criteria ranking and Pareto decision engine."""

from syndset.ranking import ModelCandidate, RankingEngine


def test_ranking_engine_veto():
  """Tests that a model with NaN or zero gradients is vetoed."""
  cand_healthy = ModelCandidate("HealthyModel", None)
  cand_healthy.raw_diagnostics = {
    "stability": {
      "status": "healthy",
      "has_nan": False,
      "has_inf": False
    },
    "gradient": {
      "status": "healthy",
      "vanishing_ratio": 0.5,
      "zero_grad_params": []
    },
    "profiling": {
      "param_count": 1000
    },
  }

  cand_unstable = ModelCandidate("UnstableModel", None)
  cand_unstable.raw_diagnostics = {
    "stability": {
      "status": "failed",
      "has_nan": True,
      "has_inf": False
    },
    "gradient": {
      "status": "healthy",
      "vanishing_ratio": 0.5,
      "zero_grad_params": []
    },
    "profiling": {
      "param_count": 1000
    },
  }

  engine = RankingEngine(profile="balanced")
  ranked = engine.rank_candidates([cand_unstable, cand_healthy])

  assert ranked[0].name == "HealthyModel"
  assert ranked[0].verdict == "PROMOTE"
  assert ranked[1].name == "UnstableModel"
  assert ranked[1].vetoed is True
  assert ranked[1].verdict == "DROP"


def test_ranking_engine_pareto_dominance():
  """Tests that a candidate dominated on all metrics is flagged."""
  cand_superior = ModelCandidate("SuperiorModel", None)
  cand_superior.raw_diagnostics = {
    "stability": {
      "status": "healthy"
    },
    "gradient": {
      "status": "healthy"
    },
    "capacity": {
      "steps_to_converge": 5,
      "converged": True
    },
    "curvature": {
      "mean_curvature": 0.1
    },
    "profiling": {
      "param_count": 500
    },
  }

  cand_dominated = ModelCandidate("DominatedModel", None)
  cand_dominated.raw_diagnostics = {
    "stability": {
      "status": "healthy"
    },
    "gradient": {
      "status": "healthy"
    },
    "capacity": {
      "steps_to_converge": 80,
      "converged": False
    },
    "curvature": {
      "mean_curvature": 50.0
    },
    "profiling": {
      "param_count": 50000
    },
  }

  engine = RankingEngine(profile="balanced")
  ranked = engine.rank_candidates([cand_dominated, cand_superior])

  assert ranked[0].name == "SuperiorModel"
  assert ranked[0].is_dominated is False
  assert ranked[1].name == "DominatedModel"
  assert ranked[1].is_dominated is True
  assert ranked[1].verdict == "DROP"
