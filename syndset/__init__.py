"""syndset: Diagnostic battery and synthetic evaluation toolkit for PyTorch neural architectures."""

__version__ = "0.1.0"

from syndset.audit import (
  audit,
  audit_llm,
  audit_tabular,
  audit_timeseries,
  audit_vision,
)
from syndset.diagnostics import (
  check_dead_units,
  check_effective_rank,
  check_gradient_flow,
  check_gradient_snr,
  check_numerical_stability,
  check_overfit_capacity,
)
from syndset.domains import (
  AssociativeRecallDataset,
  AutoregressiveLagDataset,
  ConcentricHyperspheresDataset,
  CumulativeParityDataset,
  HarmonicSuperpositionDataset,
  IllConditionedRegressionDataset,
  InductionDataset,
  SelectiveCopyDataset,
  SparseXORDataset,
  SpatialInvarianceDataset,
  SyntheticDataset,
  SyntheticShapesDataset,
  TextureVsShapeDataset,
)
from syndset.report import AuditReport

__all__ = [
  "__version__",
  "audit",
  "audit_llm",
  "audit_vision",
  "audit_tabular",
  "audit_timeseries",
  "AuditReport",
  "check_gradient_flow",
  "check_gradient_snr",
  "check_overfit_capacity",
  "check_effective_rank",
  "check_dead_units",
  "check_numerical_stability",
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
