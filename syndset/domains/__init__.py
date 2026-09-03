"""Synthetic benchmark datasets across deep learning domains."""

from syndset.domains.base import SyntheticDataset
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

__all__ = [
  "SyntheticDataset",
  "AssociativeRecallDataset",
  "InductionDataset",
  "SelectiveCopyDataset",
  "CumulativeParityDataset",
  "SyntheticShapesDataset",
  "TextureVsShapeDataset",
  "SpatialInvarianceDataset",
  "ConcentricHyperspheresDataset",
  "SparseXORDataset",
  "IllConditionedRegressionDataset",
  "HarmonicSuperpositionDataset",
  "AutoregressiveLagDataset",
]
