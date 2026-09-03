"""Architecture diagnostics and health verification probes."""

from syndset.diagnostics.capacity import check_overfit_capacity
from syndset.diagnostics.curvature import check_curvature_sharpness
from syndset.diagnostics.gradient import check_gradient_flow, check_gradient_snr
from syndset.diagnostics.initialization import check_initialization_scale
from syndset.diagnostics.permutation import check_permutation_equivariance
from syndset.diagnostics.representation import check_dead_units, check_effective_rank
from syndset.diagnostics.stability import check_numerical_stability

__all__ = [
  "check_overfit_capacity",
  "check_gradient_flow",
  "check_gradient_snr",
  "check_dead_units",
  "check_effective_rank",
  "check_numerical_stability",
  "check_initialization_scale",
  "check_curvature_sharpness",
  "check_permutation_equivariance",
]
