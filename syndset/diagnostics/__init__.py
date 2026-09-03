"""Architecture diagnostics and health verification probes."""

from syndset.diagnostics.capacity import check_overfit_capacity
from syndset.diagnostics.gradient import check_gradient_flow, check_gradient_snr
from syndset.diagnostics.representation import check_dead_units, check_effective_rank
from syndset.diagnostics.stability import check_numerical_stability

__all__ = [
  "check_overfit_capacity",
  "check_gradient_flow",
  "check_gradient_snr",
  "check_dead_units",
  "check_effective_rank",
  "check_numerical_stability",
]
