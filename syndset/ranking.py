"""Multi-criteria decision analysis and Pareto ranking engine for model comparison."""

import math


class ModelCandidate:
  """Container for a candidate architecture and its diagnostic evaluation results."""

  def __init__(self, name, model, audit_report=None, raw_diagnostics=None):
    """Initializes a candidate record.

    Args:
      name: String identifier for the candidate architecture.
      model: The torch.nn.Module or callable factory.
      audit_report: Optional AuditReport instance.
      raw_diagnostics: Optional dict of raw diagnostic returns.
    """
    self.name = name
    self.model = model
    self.audit_report = audit_report
    self.raw_diagnostics = raw_diagnostics or {}
    self.vetoed = False
    self.veto_reasons = []
    self.metrics = {}
    self.sub_scores = {}
    self.composite_score = 0.0
    self.is_dominated = False
    self.pareto_rank = 1
    self.verdict = "PROMOTE"
    self.recommendations = []

  def to_dict(self):
    """Returns a serializable summary dictionary of candidate evaluation."""
    return {
      "name": self.name,
      "vetoed": self.vetoed,
      "veto_reasons": list(self.veto_reasons),
      "composite_score": round(self.composite_score, 4),
      "is_dominated": self.is_dominated,
      "pareto_rank": self.pareto_rank,
      "verdict": self.verdict,
      "sub_scores": {
        k: round(v, 4) for k, v in self.sub_scores.items()
      },
      "recommendations": list(self.recommendations),
    }


class RankingEngine:
  """Multi-Criteria Decision Engine implementing veto gates, Pareto analysis, and utility."""

  def __init__(self, profile="balanced"):
    """Initializes the ranking engine.

    Args:
      profile: Scoring profile ('balanced', 'cloud', 'edge').
    """
    self.profile = profile

  def rank_candidates(self, candidates):
    """Executes the 4-stage decision pipeline to rank candidate models.

    Args:
      candidates: List of ModelCandidate instances with raw_diagnostics populated.

    Returns:
      List of ModelCandidate instances sorted from best to worst.
    """
    # Stage 1: Hard Veto Filter
    for cand in candidates:
      self._apply_veto_filter(cand)

    # Extract normalized numerical features
    self._extract_candidate_metrics(candidates)

    # Stage 2: Pareto Frontier Analysis
    self._compute_pareto_dominance(candidates)

    # Stage 3: Utility Scoring
    self._compute_utility_scores(candidates)

    # Stage 4: Final Verdicts
    for cand in candidates:
      self._assign_verdict(cand)

    # Sort: non-vetoed front-1 first by composite score, then dominated fronts, then vetoed
    def _sort_key(c):
      if c.vetoed:
        return (2, 999999, 0.0)
      return (0 if not c.is_dominated else 1, c.pareto_rank, -c.composite_score)

    candidates.sort(key=_sort_key)
    return candidates

  def _apply_veto_filter(self, cand):
    """Checks for fatal architectural disqualifications."""
    diag = cand.raw_diagnostics

    # 1. Numerical stability check
    stab = diag.get("stability", {})
    if stab.get("status") == "failed" or stab.get("has_nan", False) or stab.get("has_inf", False):
      cand.vetoed = True
      cand.veto_reasons.append("Fatal numerical instability: NaN or Inf activations observed.")

    # 2. Gradient flow check
    grad = diag.get("gradient", {})
    if grad.get("status") == "failed":
      if grad.get("vanishing_ratio", 1.0) < 1e-7:
        cand.vetoed = True
        cand.veto_reasons.append(
          f"Severe vanishing gradient: ratio {grad.get('vanishing_ratio', 0.0):.1e} < 1e-7.")
      if len(grad.get("zero_grad_params", [])) > 0:
        cand.vetoed = True
        cand.veto_reasons.append(
          f"Zero gradient flow in parameters: {grad.get('zero_grad_params', [])}.")

    # 3. Representation collapse
    rank = diag.get("effective_rank", {})
    if len(rank.get("collapsed_layers", [])) > 0:
      cand.vetoed = True
      cand.veto_reasons.append(
        f"Dimensional collapse in layers: {rank.get('collapsed_layers', [])}.")

    dead = diag.get("dead_units", {})
    if dead.get("status") == "failed":
      cand.vetoed = True
      cand.veto_reasons.append(
        f"Excessive dead units in layers: {dead.get('problematic_layers', [])}.")

    # 4. SynFlow disconnection
    synflow = diag.get("synflow", {})
    if synflow.get("status") == "failed" and synflow.get("total_synflow", 1.0) <= 1e-12:
      cand.vetoed = True
      cand.veto_reasons.append("Network is structurally disconnected (total SynFlow is 0.0).")

  def _extract_candidate_metrics(self, candidates):
    """Extracts scalar metric coordinates for each candidate."""
    for cand in candidates:
      diag = cand.raw_diagnostics

      # Trainability: gradient norm, curvature flatness
      curv = diag.get("curvature", {})
      mean_curv = curv.get("mean_curvature", 1.0)
      trainability = 1.0 / (1.0 + math.log1p(max(0.0, mean_curv)))

      # Expressivity: effective rank ratio, overfit capacity, and optional induction probe
      cap = diag.get("capacity", {})
      steps_to_conv = cap.get("steps_to_converge")
      converged = cap.get("converged", False)
      initial_loss = cap.get("initial_loss")
      final_loss = cap.get("final_loss")

      if converged and steps_to_conv is not None:
        cap_score = 0.5 + 0.5 * max(0.0, 1.0 - (steps_to_conv / 100.0))
      elif initial_loss is not None and final_loss is not None and initial_loss > 0:
        rel_reduction = max(0.0, min(1.0, (initial_loss - final_loss) / initial_loss))
        cap_score = 0.1 + 0.4 * rel_reduction
      else:
        cap_score = 0.1

      rank_diag = diag.get("effective_rank", {})
      rank_ratios = rank_diag.get("rank_ratios", {}).values()
      mean_rank_ratio = sum(rank_ratios) / max(1, len(rank_ratios)) if rank_ratios else 0.5

      ind_diag = diag.get("induction")
      if ind_diag is not None:
        ind_acc = float(ind_diag.get("final_accuracy", 0.0))
        ind_red = float(ind_diag.get("loss_reduction_ratio", 0.0))
        ind_score = 0.5 * max(0.0, min(1.0, ind_acc)) + 0.5 * max(0.0, min(1.0, ind_red))
        expressivity = (cap_score * 0.35) + (mean_rank_ratio * 0.35) + (ind_score * 0.30)
      else:
        expressivity = (cap_score * 0.5) + (mean_rank_ratio * 0.5)

      # Efficiency: pool-independent, monotonic transform combining params and estimated MACs
      prof = diag.get("profiling", {})
      param_count = prof.get("param_count", 1000000)
      macs = prof.get("estimated_macs", 1000000)

      cost_p = math.log10(max(1.0, float(param_count)))
      effective_macs = float(macs) if macs and macs > 0 else float(param_count)
      cost_m = math.log10(max(1.0, effective_macs))
      # Resource cost index: 1K params/MACs = 3.0, smoothly scales down in (0, 1]
      resource_cost = 0.5 * cost_p + 0.5 * cost_m
      efficiency = 1.0 / (1.0 + max(0.0, resource_cost - 3.0) / 4.0)

      # Quantization health: inverse kurtosis
      quant = diag.get("quantization", {})
      max_kurt = quant.get("max_kurtosis", 3.0)
      quant_health = 30.0 / (30.0 + max(0.0, max_kurt - 3.0))

      cand.metrics = {
        "trainability": float(trainability),
        "expressivity": float(expressivity),
        "param_count": float(param_count),
        "macs": float(macs),
        "efficiency": float(efficiency),
        "quant_health": float(quant_health),
      }

  def _compute_pareto_dominance(self, candidates, tol=0.02):
    """Computes Pareto dominance fronts across all active decision axes.

    Uses non-dominated sorting over (expressivity, trainability, efficiency, quant_health).
    Candidate A dominates candidate B if A is at least equal to B across all axes (within
    numerical tolerance `tol`) and strictly superior in at least one axis.
    """
    valid = [c for c in candidates if not c.vetoed]
    if not valid:
      return

    dominates_map = {c: [] for c in valid}
    dominated_count = {c: 0 for c in valid}

    for a in valid:
      for b in valid:
        if a is b:
          continue
        better_or_equal = (a.metrics["expressivity"] >= b.metrics["expressivity"] - tol and
                           a.metrics["trainability"] >= b.metrics["trainability"] - tol and
                           a.metrics["efficiency"] >= b.metrics["efficiency"] - tol and
                           a.metrics["quant_health"] >= b.metrics["quant_health"] - tol)
        strictly_better = (a.metrics["expressivity"] > b.metrics["expressivity"] + tol or
                           a.metrics["trainability"] > b.metrics["trainability"] + tol or
                           a.metrics["efficiency"] > b.metrics["efficiency"] + tol or
                           a.metrics["quant_health"] > b.metrics["quant_health"] + tol)
        if better_or_equal and strictly_better:
          dominates_map[a].append(b)
          dominated_count[b] += 1

    current_front = [c for c in valid if dominated_count[c] == 0]
    front_index = 1

    while current_front:
      next_front = []
      for a in current_front:
        a.pareto_rank = front_index
        a.is_dominated = (front_index > 1)
        for b in dominates_map[a]:
          dominated_count[b] -= 1
          if dominated_count[b] == 0:
            next_front.append(b)
      front_index += 1
      current_front = next_front

  def _compute_utility_scores(self, candidates):
    """Calculates weighted composite utility score based on profile."""
    if self.profile == "edge":
      weights = {
        "trainability": 0.20,
        "expressivity": 0.25,
        "efficiency": 0.35,
        "quant_health": 0.20,
      }
    elif self.profile == "cloud":
      weights = {
        "trainability": 0.35,
        "expressivity": 0.45,
        "efficiency": 0.10,
        "quant_health": 0.10,
      }
    else:  # balanced
      weights = {
        "trainability": 0.30,
        "expressivity": 0.35,
        "efficiency": 0.20,
        "quant_health": 0.15,
      }

    for cand in candidates:
      if cand.vetoed:
        cand.composite_score = 0.0
        continue

      m = cand.metrics
      cand.sub_scores = {
        "trainability": m["trainability"],
        "expressivity": m["expressivity"],
        "efficiency": m["efficiency"],
        "quant_health": m["quant_health"],
      }

      composite = (weights["trainability"] * cand.sub_scores["trainability"] +
                   weights["expressivity"] * cand.sub_scores["expressivity"] +
                   weights["efficiency"] * cand.sub_scores["efficiency"] +
                   weights["quant_health"] * cand.sub_scores["quant_health"])
      cand.composite_score = float(composite)

  def _assign_verdict(self, cand):
    """Assigns final verdict and actionable recommendations."""
    if cand.vetoed:
      cand.verdict = "DROP"
      cand.recommendations.append("Disqualified by hard veto gates. Do not train.")
      for reason in cand.veto_reasons:
        cand.recommendations.append(f"-> {reason}")
      return

    if cand.is_dominated:
      cand.verdict = "DROP"
      cand.recommendations.append(
        f"Pareto-dominated by candidate(s) in front {cand.pareto_rank - 1} "
        "with superior trade-offs.")
      return

    diag = cand.raw_diagnostics
    warnings = []
    if diag.get("quantization", {}).get("status") == "warning":
      warnings.append("High activation kurtosis; consider adding RMSNorm or smoothing before PTQ.")
    if diag.get("cka", {}).get("status") == "warning":
      warnings.append("Identified redundant consecutive layers; depth can be pruned.")
    if diag.get("curvature", {}).get("status") == "warning":
      warnings.append("Loss surface has high local curvature; recommend lower learning rate.")
    if diag.get("induction", {}).get("status") == "warning":
      warnings.append("Weak induction circuit formation; in-context copying may underperform.")
    elif diag.get("induction", {}).get("status") == "failed":
      warnings.append("Failed induction probe; architecture lacks in-context copying plasticity.")

    if warnings:
      cand.verdict = "CONDITIONALLY_VIABLE"
      cand.recommendations.extend(warnings)
    else:
      cand.verdict = "PROMOTE"
      cand.recommendations.append(
        "Pareto-optimal candidate with clean diagnostic health. Ready for training.")
