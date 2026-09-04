"""Tests for synthetic dataset generators across all domains."""

import torch

from syndset.domains.llm import (
  AssociativeRecallDataset,
  CumulativeParityDataset,
  DyckLanguageDataset,
  InductionDataset,
  MarkovLanguageDataset,
  MultiQueryAssociativeRecallDataset,
  SelectiveCopyDataset,
)
from syndset.domains.tabular import (
  ConcentricHyperspheresDataset,
  IllConditionedRegressionDataset,
  SparseXORDataset,
  TwoSpiralsDataset,
)
from syndset.domains.timeseries import (
  AutoregressiveLagDataset,
  HarmonicSuperpositionDataset,
)
from syndset.domains.vision import (
  SpatialInvarianceDataset,
  SyntheticShapesDataset,
  TextureVsShapeDataset,
)


def test_llm_associative_recall():
  """Tests key-value associative recall sequence shapes and retrieval."""
  dataset = AssociativeRecallDataset(num_samples=20, num_pairs=4, vocab_size=32, seed=42)
  assert len(dataset) == 20
  assert dataset.num_pairs == 4
  assert dataset.vocab_size == 32
  x, y = dataset[0]
  # Length: num_pairs * 2 + 1 query token
  assert x.shape[0] == 4 * 2 + 1
  assert y.dtype == torch.long


def test_llm_multi_query_associative_recall():
  """Tests multi-query associative recall generation and query counts."""
  dataset = MultiQueryAssociativeRecallDataset(
    num_samples=20, num_pairs=5, num_queries=3, vocab_size=32, seed=42
  )
  assert len(dataset) == 20
  assert dataset.num_pairs == 5
  assert dataset.num_queries == 3
  x, y = dataset[0]
  # Length: num_pairs * 2 + num_queries
  assert x.shape[0] == 5 * 2 + 3
  assert y.shape[0] == 3
  assert y.dtype == torch.long


def test_llm_induction():
  """Tests induction dataset pattern generation."""
  dataset = InductionDataset(num_samples=10, seq_len=16, vocab_size=32, seed=42)
  assert len(dataset) == 10
  assert dataset.seq_len == 16
  assert dataset.vocab_size == 32
  x, y = dataset[0]
  assert x.shape[0] == 16
  assert x[-1] == x[0]
  assert y == x[1]


def test_llm_selective_copy():
  """Tests selective copying dataset shapes."""
  dataset = SelectiveCopyDataset(num_samples=10, num_signals=3, total_len=16, seed=42)
  assert len(dataset) == 10
  assert dataset.num_signals == 3
  assert dataset.total_len == 16
  x, y = dataset[0]
  assert x.shape[0] == 16
  assert y.shape[0] == 3


def test_llm_cumulative_parity():
  """Tests cumulative parity calculation."""
  dataset = CumulativeParityDataset(num_samples=10, seq_len=8, seed=42)
  assert len(dataset) == 10
  assert dataset.seq_len == 8
  x, y = dataset[0]
  assert x.shape[0] == 8
  assert y.shape[0] == 8
  # Ensure target is binary
  assert ((y == 0) | (y == 1)).all()


def test_llm_dyck_language():
  """Tests Dyck bracket language stack target generation."""
  dataset = DyckLanguageDataset(num_samples=15, seq_len=20, num_types=2, max_depth=4, seed=42)
  assert len(dataset) == 15
  assert dataset.seq_len == 20
  assert dataset.num_types == 2
  assert dataset.max_depth == 4
  x, y = dataset[0]
  assert x.shape[0] == 20
  assert y.shape[0] == 20
  assert y.dtype == torch.long


def test_vision_synthetic_shapes():
  """Tests image generation shapes and labels."""
  dataset = SyntheticShapesDataset(num_samples=12, img_size=16, channels=1, seed=42)
  assert len(dataset) == 12
  assert dataset.img_size == 16
  assert dataset.channels == 1
  img, label = dataset[0]
  assert img.shape == (1, 16, 16)
  assert 0 <= label.item() < 4


def test_vision_texture_vs_shape():
  """Tests texture vs shape dataset output."""
  dataset = TextureVsShapeDataset(num_samples=8, img_size=16, seed=42)
  assert len(dataset) == 8
  assert dataset.img_size == 16
  img, shape_label = dataset[0]
  assert img.shape == (1, 16, 16)
  assert shape_label.item() in (0, 1)


def test_vision_spatial_invariance():
  """Tests spatial transformation pairs."""
  dataset = SpatialInvarianceDataset(num_samples=6, img_size=16, shift_pixels=2, seed=42)
  assert len(dataset) == 6
  assert dataset.img_size == 16
  assert dataset.shift_pixels == 2
  base, shifted, label = dataset[0]
  assert base.shape == (1, 16, 16)
  assert shifted.shape == (1, 16, 16)


def test_tabular_concentric_hyperspheres():
  """Tests concentric hyperspheres generation."""
  dataset = ConcentricHyperspheresDataset(num_samples=30, num_classes=3, dim=6, seed=42)
  assert len(dataset) == 30
  assert dataset.num_classes == 3
  assert dataset.dim == 6
  x, y = dataset[0]
  assert x.shape == (6,)
  assert 0 <= y.item() < 3


def test_tabular_sparse_xor():
  """Tests sparse coordinate XOR dataset."""
  dataset = SparseXORDataset(num_samples=20, total_dim=16, active_dim=3, seed=42)
  assert len(dataset) == 20
  assert dataset.total_dim == 16
  assert dataset.active_dim == 3
  x, y = dataset[0]
  assert x.shape == (16,)
  assert y.item() in (0, 1)


def test_tabular_ill_conditioned_regression():
  """Tests ill-conditioned feature matrix generation."""
  dataset = IllConditionedRegressionDataset(num_samples=20, dim=8, condition_number=1e3, seed=42)
  assert len(dataset) == 20
  assert dataset.dim == 8
  assert dataset.condition_number == 1e3
  x, y = dataset[0]
  assert x.shape == (8,)
  assert y.dim() == 0


def test_tabular_two_spirals():
  """Tests two spirals dataset generation and binary labels."""
  dataset = TwoSpiralsDataset(num_samples=24, turns=2.0, noise_std=0.02, seed=42)
  assert len(dataset) == 24
  assert dataset.turns == 2.0
  assert dataset.noise_std == 0.02
  x, y = dataset[0]
  assert x.shape == (2,)
  assert y.item() in (0, 1)


def test_timeseries_harmonic_superposition():
  """Tests harmonic waveform generation."""
  dataset = HarmonicSuperpositionDataset(num_samples=10, seq_len=32, num_harmonics=3, seed=42)
  assert len(dataset) == 10
  assert dataset.seq_len == 32
  assert dataset.num_harmonics == 3
  x, y = dataset[0]
  assert x.shape == (31, 1)
  assert y.shape == (31, 1)


def test_timeseries_autoregressive_lag():
  """Tests autoregressive lag dependency series."""
  dataset = AutoregressiveLagDataset(num_samples=10, total_steps=32, lags=(3, 8), seed=42)
  assert len(dataset) == 10
  assert dataset.total_steps == 32
  assert dataset.lags == (3, 8)
  x, y = dataset[0]
  assert x.shape == (31, 1)
  assert y.shape == (31, 1)


def test_llm_markov_language():
  """Tests Markov language dataset shapes, probabilities, and entropy."""
  dataset = MarkovLanguageDataset(
    num_samples=20, seq_len=16, vocab_size=4, order=2, alpha=1.0, seed=42
  )
  assert len(dataset) == 20
  assert dataset.vocab_size == 4
  assert dataset.order == 2
  assert dataset.seq_len == 16
  assert dataset.alpha == 1.0

  x, y = dataset[0]
  assert x.shape == (16,)
  assert y.shape == (16,)
  assert x.dtype == torch.long
  assert y.dtype == torch.long

  # Transition matrix checks
  assert dataset.transition_matrix.shape == (16, 4)
  row_sums = dataset.transition_matrix.sum(dim=-1)
  assert torch.allclose(row_sums, torch.ones(16), atol=1e-5)

  # Target probability checks
  assert dataset.target_probs.shape == (20, 16, 4)
  prob_sums = dataset.target_probs.sum(dim=-1)
  assert torch.allclose(prob_sums, torch.ones(20, 16), atol=1e-5)

  # Theoretical entropy must be strictly positive and <= ln(vocab_size)
  assert dataset.theoretical_entropy > 0.0
  assert dataset.theoretical_entropy <= torch.log(torch.tensor(4.0)).item() + 1e-5

  # Prefix grid check
  prefixes, distributions = dataset.prefix_grid
  assert prefixes.shape == (16, 2)
  assert distributions.shape == (16, 4)
  assert "Markov Language Task" in dataset.description()


def test_llm_markov_language_order_1():
  """Tests Markov language dataset with order 1 (bigram)."""
  dataset = MarkovLanguageDataset(
    num_samples=10, seq_len=8, vocab_size=6, order=1, alpha=0.5, seed=123
  )
  assert dataset.order == 1
  assert dataset.transition_matrix.shape == (6, 6)
  prefixes, dists = dataset.prefix_grid
  assert prefixes.shape == (6, 1)
  assert dists.shape == (6, 6)


def test_llm_markov_language_evaluation():
  """Tests evaluation on prefix grid and causal leakage checking."""
  dataset = MarkovLanguageDataset(
    num_samples=20, seq_len=16, vocab_size=4, order=2, alpha=1.0, seed=42
  )

  class OracleModel(torch.nn.Module):
    def __init__(self, trans_matrix, vocab_size, order):
      super().__init__()
      self.weights = vocab_size ** torch.arange(order - 1, -1, -1, dtype=torch.long)
      self.logits = torch.log(trans_matrix + 1e-12)

    def forward(self, x):
      if x.dim() == 2:
        ctx = x[:, -2:]
        idx = (ctx * self.weights).sum(dim=-1)
        return self.logits[idx]
      return self.logits[0].expand(x.shape[0], -1)

  oracle = OracleModel(dataset.transition_matrix, dataset.vocab_size, dataset.order)
  eval_res = dataset.evaluate_distribution(oracle)
  assert eval_res["status"] == "passed"
  assert eval_res["mean_tv_distance"] < 0.05
  assert eval_res["mean_kl_divergence"] < 0.05

  leak_res = dataset.check_causal_leakage(oracle)
  assert leak_res["status"] == "causally_sound"
  assert not leak_res["is_leaking"]

  # A cheating model that produces 99% confident prediction on the target token
  class CheatingModel(torch.nn.Module):
    def __init__(self, targets, vocab_size):
      super().__init__()
      self.targets = targets
      self.vocab_size = vocab_size

    def forward(self, x):
      # Cheat by looking up the actual target for the last token
      B = x.shape[0]
      logits = torch.full((B, self.vocab_size), -10.0)
      logits.scatter_(1, self.targets[:B, -1].unsqueeze(-1), 10.0)
      return logits

  cheating = CheatingModel(dataset.targets, dataset.vocab_size)
  leak_res_cheating = dataset.check_causal_leakage(cheating)
  assert leak_res_cheating["status"] == "leaked"
  assert leak_res_cheating["is_leaking"]
