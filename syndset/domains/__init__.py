"""Synthetic benchmark datasets across deep learning domains."""

from syndset.domains.base import SyntheticDataset
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
  SHAPE_NAMES,
  SpatialInvarianceDataset,
  SyntheticDetectionDataset,
  SyntheticSegmentationDataset,
  SyntheticShapesDataset,
  TextureVsShapeDataset,
  rasterize_shape,
)

__all__ = [
  "SyntheticDataset",
  "AssociativeRecallDataset",
  "MultiQueryAssociativeRecallDataset",
  "InductionDataset",
  "MarkovLanguageDataset",
  "SelectiveCopyDataset",
  "CumulativeParityDataset",
  "DyckLanguageDataset",
  "SyntheticShapesDataset",
  "SyntheticSegmentationDataset",
  "SyntheticDetectionDataset",
  "TextureVsShapeDataset",
  "SpatialInvarianceDataset",
  "ConcentricHyperspheresDataset",
  "SparseXORDataset",
  "IllConditionedRegressionDataset",
  "TwoSpiralsDataset",
  "HarmonicSuperpositionDataset",
  "AutoregressiveLagDataset",
  "SHAPE_NAMES",
  "rasterize_shape",
]
