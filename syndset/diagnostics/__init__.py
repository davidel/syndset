"""Architecture diagnostics and health verification probes."""

from syndset.diagnostics.attribution import (
  check_attribution_faithfulness,
  check_context_attribution_decay,
  check_layer_conductance,
  compute_integrated_gradients,
)
from syndset.diagnostics.capacity import check_overfit_capacity
from syndset.diagnostics.cka import check_layer_cka, compute_linear_cka
from syndset.diagnostics.curvature import check_curvature_sharpness
from syndset.diagnostics.erf import check_effective_receptive_field
from syndset.diagnostics.gradient import check_gradient_flow, check_gradient_snr
from syndset.diagnostics.induction import check_induction_capability
from syndset.diagnostics.initialization import check_initialization_scale
from syndset.diagnostics.ntk import check_ntk_conditioning
from syndset.diagnostics.permutation import check_permutation_equivariance
from syndset.diagnostics.profiling import estimate_module_macs, profile_model_efficiency
from syndset.diagnostics.quantization import check_quantization_health, compute_sample_kurtosis
from syndset.diagnostics.representation import check_dead_units, check_effective_rank
from syndset.diagnostics.stability import check_numerical_stability
from syndset.diagnostics.synflow import check_synflow

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
  "compute_integrated_gradients",
  "check_attribution_faithfulness",
  "check_layer_conductance",
  "check_context_attribution_decay",
  "check_layer_cka",
  "compute_linear_cka",
  "check_synflow",
  "check_quantization_health",
  "compute_sample_kurtosis",
  "check_ntk_conditioning",
  "check_induction_capability",
  "check_effective_receptive_field",
  "profile_model_efficiency",
  "estimate_module_macs",
]
