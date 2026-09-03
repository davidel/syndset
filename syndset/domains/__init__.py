"""Synthetic benchmark datasets across deep learning domains."""

from syndset.domains.base import SyntheticDataset
from syndset.domains.llm import (
  AssociativeRecallDataset,
  CumulativeParityDataset,
  DyckLanguageDataset,
  InductionDataset,
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

__all__ = [
  "SyntheticDataset",
  "AssociativeRecallDataset",
  "MultiQueryAssociativeRecallDataset",
  "InductionDataset",
  "SelectiveCopyDataset",
  "CumulativeParityDataset",
  "DyckLanguageDataset",
  "SyntheticShapesDataset",
  "TextureVsShapeDataset",
  "SpatialInvarianceDataset",
  "ConcentricHyperspheresDataset",
  "SparseXORDataset",
  "IllConditionedRegressionDataset",
  "TwoSpiralsDataset",
  "HarmonicSuperpositionDataset",
  "AutoregressiveLagDataset",
]
