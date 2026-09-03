"""Tests for synthetic dataset generators across all domains."""

import torch

from syndset.domains.llm import (
  AssociativeRecallDataset,
  CumulativeParityDataset,
  InductionDataset,
  SelectiveCopyDataset,
)
from syndset.domains.tabular import (
  ConcentricHyperspheresDataset,
  IllConditionedRegressionDataset,
  SparseXORDataset,
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
  x, y = dataset[0]
  # Length: num_pairs * 2 + 1 query token
  assert x.shape[0] == 4 * 2 + 1
  assert y.dtype == torch.long


def test_llm_induction():
  """Tests induction dataset pattern generation."""
  dataset = InductionDataset(num_samples=10, seq_len=16, vocab_size=32, seed=42)
  assert len(dataset) == 10
  x, y = dataset[0]
  assert x.shape[0] == 16
  assert x[-1] == x[0]
  assert y == x[1]


def test_llm_selective_copy():
  """Tests selective copying dataset shapes."""
  dataset = SelectiveCopyDataset(num_samples=10, num_signals=3, total_len=16, seed=42)
  assert len(dataset) == 10
  x, y = dataset[0]
  assert x.shape[0] == 16
  assert y.shape[0] == 3


def test_llm_cumulative_parity():
  """Tests cumulative parity calculation."""
  dataset = CumulativeParityDataset(num_samples=10, seq_len=8, seed=42)
  assert len(dataset) == 10
  x, y = dataset[0]
  assert x.shape[0] == 8
  assert y.shape[0] == 8
  # Ensure target is binary
  assert ((y == 0) | (y == 1)).all()


def test_vision_synthetic_shapes():
  """Tests image generation shapes and labels."""
  dataset = SyntheticShapesDataset(num_samples=12, img_size=16, channels=1, seed=42)
  assert len(dataset) == 12
  img, label = dataset[0]
  assert img.shape == (1, 16, 16)
  assert 0 <= label.item() < 4


def test_vision_texture_vs_shape():
  """Tests texture vs shape dataset output."""
  dataset = TextureVsShapeDataset(num_samples=8, img_size=16, seed=42)
  assert len(dataset) == 8
  img, shape_label = dataset[0]
  assert img.shape == (1, 16, 16)
  assert shape_label.item() in (0, 1)


def test_vision_spatial_invariance():
  """Tests spatial transformation pairs."""
  dataset = SpatialInvarianceDataset(num_samples=6, img_size=16, shift_pixels=2, seed=42)
  assert len(dataset) == 6
  base, shifted, label = dataset[0]
  assert base.shape == (1, 16, 16)
  assert shifted.shape == (1, 16, 16)


def test_tabular_concentric_hyperspheres():
  """Tests concentric hyperspheres generation."""
  dataset = ConcentricHyperspheresDataset(num_samples=30, num_classes=3, dim=6, seed=42)
  assert len(dataset) == 30
  x, y = dataset[0]
  assert x.shape == (6,)
  assert 0 <= y.item() < 3


def test_tabular_sparse_xor():
  """Tests sparse coordinate XOR dataset."""
  dataset = SparseXORDataset(num_samples=20, total_dim=16, active_dim=3, seed=42)
  assert len(dataset) == 20
  x, y = dataset[0]
  assert x.shape == (16,)
  assert y.item() in (0, 1)


def test_tabular_ill_conditioned_regression():
  """Tests ill-conditioned feature matrix generation."""
  dataset = IllConditionedRegressionDataset(num_samples=20, dim=8, condition_number=1e3, seed=42)
  assert len(dataset) == 20
  x, y = dataset[0]
  assert x.shape == (8,)
  assert y.dim() == 0


def test_timeseries_harmonic_superposition():
  """Tests harmonic waveform generation."""
  dataset = HarmonicSuperpositionDataset(num_samples=10, seq_len=32, num_harmonics=3, seed=42)
  assert len(dataset) == 10
  x, y = dataset[0]
  assert x.shape == (31, 1)
  assert y.shape == (31, 1)


def test_timeseries_autoregressive_lag():
  """Tests autoregressive lag dependency series."""
  dataset = AutoregressiveLagDataset(num_samples=10, total_steps=32, lags=(3, 8), seed=42)
  assert len(dataset) == 10
  x, y = dataset[0]
  assert x.shape == (31, 1)
  assert y.shape == (31, 1)
