"""Synthetic benchmark datasets across deep learning domains."""

from syndset.domains.audio import (
  FMChirpDataset,
  PolyphonicSoundEventDataset,
)
from syndset.domains.base import (
  DataSample,
  SyntheticDataset,
  list_datasets,
  load_dataset,
  register_dataset,
)
from syndset.domains.continual import (
  RotatingManifoldStreamDataset,
  SpuriousCorrelationDataset,
)
from syndset.domains.generative import (
  AnalyticalScoreDataset,
  FlowMatchingDataset,
  ToyManifold2DDataset,
)
from syndset.domains.graphs import (
  SE3EquivarianceDataset,
  ShortestPathDataset,
  WLExpressivityDataset,
)
from syndset.domains.interpretability import (
  FeatureSuperpositionDataset,
  ModularArithmeticDataset,
)
from syndset.domains.llm import (
  AssociativeRecallDataset,
  CumulativeParityDataset,
  DyckLanguageDataset,
  InductionDataset,
  MarkovLanguageDataset,
  MultiQueryAssociativeRecallDataset,
  NeedleInHaystackDataset,
  ReasoningDAGDataset,
  SelectiveCopyDataset,
  StructuredGrammarDataset,
)
from syndset.domains.multimodal import (
  MultimodalShapesDataset,
  VisualQuestionAnsweringDataset,
)
from syndset.domains.neural_fields import (
  AnalyticalSDFDataset,
  ToyNeRFDataset,
)
from syndset.domains.odometry import (
  OpticalFlowDataset,
  VisualOdometryDataset,
)
from syndset.domains.physics import (
  BurgersEquationDataset,
  HeatEquationDataset,
)
from syndset.domains.rl import (
  CartPoleMDP,
  CartPoleOfflineDataset,
  GridworldDataset,
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
  ArticulatedKeypointsDataset,
  NumerosityDataset,
  OccludedDetectionDataset,
  SpatialInvarianceDataset,
  SpatialRelationDataset,
  SyntheticDetectionDataset,
  SyntheticSegmentationDataset,
  SyntheticShapesDataset,
  TextureVsShapeDataset,
  rasterize_shape,
)

__all__ = [
  "DataSample",
  "SyntheticDataset",
  "list_datasets",
  "load_dataset",
  "register_dataset",
  # Multimodal
  "MultimodalShapesDataset",
  "VisualQuestionAnsweringDataset",
  # Odometry
  "VisualOdometryDataset",
  "OpticalFlowDataset",
  # RL
  "CartPoleMDP",
  "CartPoleOfflineDataset",
  "GridworldDataset",
  # Audio
  "FMChirpDataset",
  "PolyphonicSoundEventDataset",
  # Generative
  "AnalyticalScoreDataset",
  "FlowMatchingDataset",
  "ToyManifold2DDataset",
  # Graphs
  "WLExpressivityDataset",
  "ShortestPathDataset",
  "SE3EquivarianceDataset",
  # Neural Fields
  "AnalyticalSDFDataset",
  "ToyNeRFDataset",
  # Interpretability
  "ModularArithmeticDataset",
  "FeatureSuperpositionDataset",
  # Physics
  "HeatEquationDataset",
  "BurgersEquationDataset",
  # Continual
  "RotatingManifoldStreamDataset",
  "SpuriousCorrelationDataset",
  # LLM
  "AssociativeRecallDataset",
  "MultiQueryAssociativeRecallDataset",
  "InductionDataset",
  "MarkovLanguageDataset",
  "SelectiveCopyDataset",
  "CumulativeParityDataset",
  "DyckLanguageDataset",
  "NeedleInHaystackDataset",
  "StructuredGrammarDataset",
  "ReasoningDAGDataset",
  # Vision
  "SyntheticShapesDataset",
  "SyntheticSegmentationDataset",
  "SyntheticDetectionDataset",
  "NumerosityDataset",
  "SpatialRelationDataset",
  "TextureVsShapeDataset",
  "SpatialInvarianceDataset",
  "OccludedDetectionDataset",
  "ArticulatedKeypointsDataset",
  "SHAPE_NAMES",
  "rasterize_shape",
  # Tabular
  "ConcentricHyperspheresDataset",
  "SparseXORDataset",
  "IllConditionedRegressionDataset",
  "TwoSpiralsDataset",
  # Time series
  "HarmonicSuperpositionDataset",
  "AutoregressiveLagDataset",
]
