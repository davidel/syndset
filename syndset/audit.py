"""Top-level architecture audit orchestrator."""

from syndset.diagnostics.capacity import check_overfit_capacity
from syndset.diagnostics.gradient import check_gradient_flow
from syndset.diagnostics.representation import check_dead_units, check_effective_rank
from syndset.diagnostics.stability import check_numerical_stability
from syndset.domains.llm import AssociativeRecallDataset
from syndset.domains.tabular import ConcentricHyperspheresDataset
from syndset.domains.timeseries import HarmonicSuperpositionDataset
from syndset.domains.vision import SyntheticShapesDataset
from syndset.report import AuditReport


def audit(
  model,
  sample_input,
  sample_target=None,
  loss_fn=None,
  run_overfit=True,
  run_rank=True,
  run_stability=True,
  model_name=None,
):
  """Runs a comprehensive battery of diagnostic health checks on a neural architecture.

  Executes gradient flow verification, representation collapse checks,
  numerical stability tests, and single-batch capacity tests in seconds.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: Sample tensor or dict/tuple matching model forward inputs.
    sample_target: Optional target tensor corresponding to sample_input.
    loss_fn: Optional loss function.
    run_overfit: Whether to run single-batch memorization test (default: True).
    run_rank: Whether to compute effective rank and dead units (default: True).
    run_stability: Whether to check half precision and NaN stability (default: True).
    model_name: Optional name for the model in report summary.

  Returns:
    An AuditReport instance containing detailed findings and scorecard.
  """
  name = model_name or model.__class__.__name__
  report = AuditReport(model_name=name)

  # 1. Gradient Flow Probe
  grad_res = check_gradient_flow(model, sample_input, target=sample_target, loss_fn=loss_fn)
  num_zeros = len(grad_res["zero_grad_params"])
  summary_text = (
    f"Norm: {grad_res['global_norm']:.3f}, "
    f"Vanishing ratio: {grad_res['vanishing_ratio']:.2e}, "
    f"Zero-grad params: {num_zeros}"
  )
  report.add_check("Gradient Flow", grad_res["status"], summary_text, grad_res["issues"])

  # 2. Representation & Rank Probes
  if run_rank:
    rank_res = check_effective_rank(model, sample_input)
    num_collapsed = len(rank_res["collapsed_layers"])
    rank_summary = f"{len(rank_res['layer_ranks'])} layers monitored, {num_collapsed} collapsed"
    report.add_check("Effective Rank", rank_res["status"], rank_summary, rank_res["issues"])

    dead_res = check_dead_units(model, sample_input)
    num_dead_layers = len(dead_res["problematic_layers"])
    dead_summary = f"{num_dead_layers} layers with >30% dead units"
    report.add_check("Dead Units", dead_res["status"], dead_summary, dead_res["issues"])

  # 3. Numerical Stability Probe
  if run_stability:
    stab_res = check_numerical_stability(model, sample_input)
    half_err = stab_res["half_precision_relative_error"]
    err_str = f"{half_err:.2%}" if half_err is not None else "N/A"
    stab_summary = f"Max act: {stab_res['max_abs_activation']:.2f}, Half diff: {err_str}"
    report.add_check("Numerical Stability", stab_res["status"], stab_summary, stab_res["issues"])

  # 4. Capacity & Single-Batch Overfit Probe
  if run_overfit and sample_target is not None:
    cap_res = check_overfit_capacity(model, sample_input, sample_target, loss_fn=loss_fn)
    steps_val = cap_res["steps_to_converge"]
    steps_str = f"in {steps_val} steps" if cap_res["converged"] else "not converged"
    cap_summary = f"Final loss: {cap_res['final_loss']:.4f} ({steps_str})"
    report.add_check("Overfit Capacity", cap_res["status"], cap_summary, cap_res["issues"])

  return report


def audit_llm(model, num_pairs=8, vocab_size=64, batch_size=8, seed=42):
  """Audits an autoregressive or sequence model on synthetic associative recall.

  Args:
    model: The sequence torch.nn.Module to evaluate.
    num_pairs: Number of key-value pairs in sequence (default: 8).
    vocab_size: Vocabulary size (default: 64).
    batch_size: Batch size for audit inputs (default: 8).
    seed: Random seed (default: 42).

  Returns:
    An AuditReport summarizing model health on the sequence task.
  """
  dataset = AssociativeRecallDataset(
    num_samples=batch_size, num_pairs=num_pairs, vocab_size=vocab_size, seed=seed
  )
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (LLM Task)")


def audit_vision(model, img_size=32, channels=1, batch_size=8, seed=42):
  """Audits a vision model on procedural geometric shapes.

  Args:
    model: The vision torch.nn.Module to evaluate.
    img_size: Image height and width (default: 32).
    channels: Image channels (default: 1).
    batch_size: Batch size for audit (default: 8).
    seed: Random seed (default: 42).

  Returns:
    An AuditReport summarizing model health on synthetic vision inputs.
  """
  dataset = SyntheticShapesDataset(
    num_samples=batch_size, img_size=img_size, channels=channels, seed=seed
  )
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (Vision Task)")


def audit_tabular(model, dim=16, num_classes=3, batch_size=16, seed=42):
  """Audits a feedforward / tabular architecture on concentric hyperspheres.

  Args:
    model: The tabular torch.nn.Module to evaluate.
    dim: Dimensionality of inputs (default: 16).
    num_classes: Number of concentric spherical shells (default: 3).
    batch_size: Batch size for audit (default: 16).
    seed: Random seed (default: 42).

  Returns:
    An AuditReport summarizing model health on non-linear manifold inputs.
  """
  dataset = ConcentricHyperspheresDataset(
    num_samples=batch_size, num_classes=num_classes, dim=dim, seed=seed
  )
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (Tabular Task)")


def audit_timeseries(model, seq_len=32, batch_size=8, seed=42):
  """Audits a time series or recurrent architecture on harmonic waveforms.

  Args:
    model: The time-series torch.nn.Module to evaluate.
    seq_len: Number of time steps (default: 32).
    batch_size: Batch size for audit (default: 8).
    seed: Random seed (default: 42).

  Returns:
    An AuditReport summarizing model health on spectral wave inputs.
  """
  dataset = HarmonicSuperpositionDataset(num_samples=batch_size, seq_len=seq_len, seed=seed)
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (Time Series Task)")
