"""Multi-model tournament orchestration and comparative architecture evaluation."""

import torch

from syndset.audit import audit
from syndset.diagnostics.capacity import check_overfit_capacity
from syndset.diagnostics.cka import check_layer_cka
from syndset.diagnostics.curvature import check_curvature_sharpness
from syndset.diagnostics.erf import check_effective_receptive_field
from syndset.diagnostics.gradient import check_gradient_flow
from syndset.diagnostics.induction import check_induction_capability
from syndset.diagnostics.profiling import profile_model_efficiency
from syndset.diagnostics.quantization import check_quantization_health
from syndset.diagnostics.representation import check_dead_units, check_effective_rank
from syndset.diagnostics.stability import check_numerical_stability
from syndset.diagnostics.synflow import check_synflow
from syndset.domains.llm import AssociativeRecallDataset
from syndset.domains.tabular import ConcentricHyperspheresDataset
from syndset.domains.timeseries import HarmonicSuperpositionDataset
from syndset.domains.vision import SyntheticShapesDataset
from syndset.ranking import ModelCandidate, RankingEngine


class TournamentReport:
  """Aggregates and renders multi-model tournament outcomes and rankings."""

  def __init__(self, candidates, domain="general", profile="balanced"):
    """Initializes the tournament report.

    Args:
      candidates: List of ranked ModelCandidate instances.
      domain: String domain descriptor.
      profile: Decision profile name.
    """
    self.candidates = candidates
    self.domain = domain
    self.profile = profile

  def to_dict(self):
    """Returns a serializable dictionary representation of tournament results."""
    return {
      "domain": self.domain,
      "profile": self.profile,
      "rankings": [c.to_dict() for c in self.candidates],
    }

  def summary(self):
    """Generates a formatted multi-line string table ranking all candidates."""
    lines = []
    width = 100
    sep = "=" * width
    lines.append(sep)
    lines.append(
      f"  SYNDSET ARCHITECTURE TOURNAMENT: {self.domain.upper()} (Profile: {self.profile})")
    lines.append(sep)

    header = (f"{'Rank':<5} | {'Model':<20} | {'Score':<7} | {'Train':<6} | {'Expr':<6} | "
              f"{'Eff':<6} | {'Quant':<6} | {'Verdict':<15} | {'Status'}")
    lines.append(header)
    lines.append("-" * width)

    for rank_idx, cand in enumerate(self.candidates, start=1):
      tag = "[VETO]" if cand.vetoed else ("[PARETO]" if not cand.is_dominated else "[DOMINATED]")
      score_str = f"{cand.composite_score:.3f}" if not cand.vetoed else " N/A "
      subs = cand.sub_scores
      t_str = f"{subs.get('trainability', 0.0):.2f}" if not cand.vetoed else " - "
      e_str = f"{subs.get('expressivity', 0.0):.2f}" if not cand.vetoed else " - "
      eff_str = f"{subs.get('efficiency', 0.0):.2f}" if not cand.vetoed else " - "
      q_str = f"{subs.get('quant_health', 0.0):.2f}" if not cand.vetoed else " - "
      verdict_icon = ("🟢 PROMOTE" if cand.verdict == "PROMOTE" else
                      ("🟡 CONDITIONAL" if cand.verdict == "CONDITIONALLY_VIABLE" else "🔴 DROP"))
      lines.append(f"{rank_idx:<5} | {cand.name:<20} | {score_str:<7} | {t_str:<6} | {e_str:<6} | "
                   f"{eff_str:<6} | {q_str:<6} | {verdict_icon:<15} | {tag}")
      for rec in cand.recommendations:
        lines.append(f"       -> {rec}")

    lines.append(sep)

    # Flag non-discriminating metric components across active candidates
    active_cands = [c for c in self.candidates if not c.vetoed]
    if len(active_cands) >= 2:
      non_disc = []
      for metric_key in ("trainability", "expressivity", "efficiency", "quant_health"):
        vals = [c.sub_scores.get(metric_key, 0.0) for c in active_cands]
        if max(vals) - min(vals) < 1e-4:
          non_disc.append(metric_key)
      if non_disc:
        lines.append(f"  Note: Non-discriminating metrics across candidates: {', '.join(non_disc)}")
        lines.append(sep)

    return "\n".join(lines)

  def print_summary(self):
    """Prints formatted scorecard directly to standard output."""
    print(self.summary())


class Tournament:
  """Orchestrator for multi-model diagnostic tournaments."""

  def __init__(self, domain="general", profile="balanced", domain_kwargs=None):
    """Initializes a tournament.

    Args:
      domain: Target domain ('general', 'llm', 'vision', 'tabular', 'timeseries').
      profile: Scoring profile ('balanced', 'cloud', 'edge').
      domain_kwargs: Optional domain-specific keyword arguments.
    """
    self.domain = domain
    self.profile = profile
    self.domain_kwargs = domain_kwargs or {}
    self.ranking_engine = RankingEngine(profile=profile)

  def evaluate(
    self,
    candidates,
    sample_input,
    sample_target=None,
    loss_fn=None,
    run_curvature=True,
    run_synflow=True,
    run_quantization=True,
    run_cka=True,
    run_induction=True,
    capacity_num_steps=50,
  ):
    """Evaluates a pool of candidate architectures on identical synthetic inputs.

    Args:
      candidates: Dict mapping model_name to torch.nn.Module, or list of nn.Modules.
      sample_input: Sample input tensor or tuple/dict matching forward inputs.
      sample_target: Optional target tensor for overfit and capacity checks.
      loss_fn: Optional loss callable.
      run_curvature: Whether to evaluate loss surface curvature (default: True).
      run_synflow: Whether to evaluate SynFlow parameter saliency (default: True).
      run_quantization: Whether to evaluate activation kurtosis for PTQ (default: True).
      run_cka: Whether to evaluate layer CKA similarity (default: True).
      run_induction: Whether to evaluate LLM in-context induction capability (default: True).
      capacity_num_steps: Optimization step budget for capacity probe (default: 50).

    Returns:
      A TournamentReport instance containing ranked candidates and scorecards.
    """
    # Normalize candidates to list of (name, model)
    model_pairs = []
    if isinstance(candidates, dict):
      model_pairs = list(candidates.items())
    elif isinstance(candidates, (list, tuple)):
      for idx, item in enumerate(candidates):
        if isinstance(item, tuple) and len(item) == 2:
          model_pairs.append(item)
        else:
          name = getattr(item, "__class__", type(item)).__name__
          model_pairs.append((f"{name}_{idx+1}", item))

    candidate_records = []

    for name, model in model_pairs:
      # Run base audit (skipping overfit here to avoid duplicate execution)
      audit_rep = audit(
        model,
        sample_input,
        sample_target=sample_target,
        loss_fn=loss_fn,
        run_overfit=False,
        model_name=name,
      )

      # Collect raw diagnostic dictionaries
      raw_diags = {}
      raw_diags["gradient"] = check_gradient_flow(model,
                                                  sample_input,
                                                  target=sample_target,
                                                  loss_fn=loss_fn)
      raw_diags["effective_rank"] = check_effective_rank(model, sample_input)
      raw_diags["dead_units"] = check_dead_units(model, sample_input)
      raw_diags["stability"] = check_numerical_stability(model, sample_input)
      raw_diags["profiling"] = profile_model_efficiency(model, sample_input)

      # Capacity probe with customizable step budget for tournament efficiency
      if sample_target is not None and capacity_num_steps > 0:
        raw_diags["capacity"] = check_overfit_capacity(
          model,
          sample_input,
          sample_target,
          loss_fn=loss_fn,
          num_steps=capacity_num_steps,
        )
        cap_res = raw_diags["capacity"]
        steps_val = cap_res["steps_to_converge"]
        steps_str = f"in {steps_val} steps" if cap_res["converged"] else "not converged"
        cap_summary = f"Final loss: {cap_res['final_loss']:.4f} ({steps_str})"
        audit_rep.add_check("Overfit Capacity", cap_res["status"], cap_summary, cap_res["issues"])

      if run_curvature:
        raw_diags["curvature"] = check_curvature_sharpness(model,
                                                           sample_input,
                                                           target=sample_target,
                                                           loss_fn=loss_fn)
      if run_synflow:
        raw_diags["synflow"] = check_synflow(model, sample_input)
      if run_quantization:
        raw_diags["quantization"] = check_quantization_health(model, sample_input)
      if run_cka:
        raw_diags["cka"] = check_layer_cka(model, sample_input)

      # Domain-specific micro-probes
      if self.domain == "llm" and run_induction:
        vocab_limit = self.domain_kwargs.get("vocab_size", 64)
        raw_diags["induction"] = check_induction_capability(model, vocab_size=vocab_limit)
        ind_res = raw_diags["induction"]
        ind_summary = (f"Acc: {ind_res['final_accuracy']:.0%}, "
                       f"Loss red: {ind_res['loss_reduction_ratio']:.1%}")
        audit_rep.add_check(
          "Induction Capability",
          ind_res["status"],
          ind_summary,
          ind_res["issues"],
        )
      elif (self.domain == "vision" and isinstance(sample_input, torch.Tensor) and
            sample_input.dim() == 4):
        raw_diags["erf"] = check_effective_receptive_field(model, input_shape=sample_input.shape)

      record = ModelCandidate(
        name=name,
        model=model,
        audit_report=audit_rep,
        raw_diagnostics=raw_diags,
      )
      candidate_records.append(record)

    ranked_records = self.ranking_engine.rank_candidates(candidate_records)
    return TournamentReport(ranked_records, domain=self.domain, profile=self.profile)


def tournament_llm(
  candidates,
  num_pairs=8,
  vocab_size=64,
  batch_size=8,
  profile="balanced",
  seed=42,
):
  """Executes an architecture tournament across candidate LLM / sequence models.

  Candidate models must accept input token ID tensors of shape `(batch_size, 2 * num_pairs + 1)`.
  If a model outputs 3D logits `(batch_size, seq_len, vocab_size)`, syndset automatically
  slices the final sequence token `[:, -1, :]` for next-token prediction evaluation.

  Args:
    candidates: Dict or list of sequence torch.nn.Modules.
    num_pairs: Distinct key-value pairs per sequence (default: 8).
    vocab_size: Total vocabulary size (default: 64).
    batch_size: Evaluation batch size (default: 8).
    profile: Scoring profile ('balanced', 'cloud', 'edge').
    seed: Seed for reproducible synthetic task (default: 42).

  Returns:
    A TournamentReport with ranked LLM candidates.
  """
  dataset = AssociativeRecallDataset(
    num_samples=batch_size,
    num_pairs=num_pairs,
    vocab_size=vocab_size,
    seed=seed,
  )
  inputs, targets = dataset[:batch_size]
  tourney = Tournament(domain="llm", profile=profile, domain_kwargs={"vocab_size": vocab_size})
  return tourney.evaluate(candidates, inputs, targets)


def tournament_vision(
  candidates,
  img_size=32,
  channels=1,
  num_classes=8,
  batch_size=8,
  profile="balanced",
  seed=42,
):
  """Executes an architecture tournament across candidate vision classifiers.

  Args:
    candidates: Dict or list of vision torch.nn.Modules.
    img_size: Height/width in pixels (default: 32).
    channels: Image channels (default: 1).
    num_classes: Number of shape classes (default: 8).
    batch_size: Evaluation batch size (default: 8).
    profile: Scoring profile ('balanced', 'cloud', 'edge').
    seed: Seed for reproducible synthetic task (default: 42).

  Returns:
    A TournamentReport with ranked vision candidates.
  """
  dataset = SyntheticShapesDataset(
    num_samples=batch_size,
    img_size=img_size,
    channels=channels,
    num_classes=num_classes,
    seed=seed,
  )
  inputs, targets = dataset[:batch_size]
  tourney = Tournament(domain="vision", profile=profile)
  return tourney.evaluate(candidates, inputs, targets)


def tournament_tabular(
  candidates,
  dim=16,
  num_classes=3,
  batch_size=16,
  profile="balanced",
  seed=42,
):
  """Executes an architecture tournament across candidate tabular / MLP architectures.

  Args:
    candidates: Dict or list of tabular torch.nn.Modules.
    dim: Input feature dimensionality (default: 16).
    num_classes: Number of concentric spherical classes (default: 3).
    batch_size: Evaluation batch size (default: 16).
    profile: Scoring profile ('balanced', 'cloud', 'edge').
    seed: Seed for reproducible synthetic task (default: 42).

  Returns:
    A TournamentReport with ranked tabular candidates.
  """
  dataset = ConcentricHyperspheresDataset(
    num_samples=batch_size,
    num_classes=num_classes,
    dim=dim,
    seed=seed,
  )
  inputs, targets = dataset[:batch_size]
  tourney = Tournament(domain="tabular", profile=profile)
  return tourney.evaluate(candidates, inputs, targets)


def tournament_timeseries(candidates, seq_len=32, batch_size=8, profile="balanced", seed=42):
  """Executes an architecture tournament across candidate time-series models.

  Args:
    candidates: Dict or list of time-series torch.nn.Modules.
    seq_len: Waveform length (default: 32).
    batch_size: Evaluation batch size (default: 8).
    profile: Scoring profile ('balanced', 'cloud', 'edge').
    seed: Seed for reproducible synthetic task (default: 42).

  Returns:
    A TournamentReport with ranked time-series candidates.
  """
  dataset = HarmonicSuperpositionDataset(
    num_samples=batch_size,
    seq_len=seq_len,
    seed=seed,
  )
  inputs, targets = dataset[:batch_size]
  tourney = Tournament(domain="timeseries", profile=profile)
  return tourney.evaluate(candidates, inputs, targets)
