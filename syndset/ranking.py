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

    # Stage 3: Utility Scoring (TOPSIS-inspired)
    self._compute_utility_scores(candidates)

    # Stage 4: Final Verdicts
    for cand in candidates:
      self._assign_verdict(cand)

    # Sort: non-vetoed first (by composite_score desc), vetoed last
    def _sort_key(c):
      return (0 if not c.vetoed else 1, -c.composite_score)

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

      # Expressivity: effective rank ratio, overfit capacity
      cap = diag.get("capacity", {})
      steps_to_conv = cap.get("steps_to_converge")
      converged = cap.get("converged", False)
      initial_loss = cap.get("initial_loss")
      final_loss = cap.get("final_loss")

      if converged and steps_to_conv is not None:
        cap_score = 0.5 + 0.5 * max(0.0, 1.0 - (steps_to_conv / 100.0))
      elif initial_loss is not None and final_loss is not None and initial_loss > 0:
        # Continuous loss reduction provides signal even when hard convergence threshold is unmet
        rel_reduction = max(0.0, min(1.0, (initial_loss - final_loss) / initial_loss))
        cap_score = 0.1 + 0.4 * rel_reduction
      else:
        cap_score = 0.1

      rank_diag = diag.get("effective_rank", {})
      rank_ratios = rank_diag.get("rank_ratios", {}).values()
      mean_rank_ratio = sum(rank_ratios) / max(1, len(rank_ratios)) if rank_ratios else 0.5
      expressivity = (cap_score * 0.5) + (mean_rank_ratio * 0.5)

      # Efficiency: inverse param count and estimated MACs
      prof = diag.get("profiling", {})
      param_count = prof.get("param_count", 1000000)
      macs = prof.get("estimated_macs", 1000000)

      # Quantization health: inverse kurtosis
      quant = diag.get("quantization", {})
      max_kurt = quant.get("max_kurtosis", 3.0)
      quant_health = 30.0 / (30.0 + max(0.0, max_kurt - 3.0))

      cand.metrics = {
        "trainability": float(trainability),
        "expressivity": float(expressivity),
        "param_count": float(param_count),
        "macs": float(macs),
        "quant_health": float(quant_health),
      }

  def _compute_pareto_dominance(self, candidates, tol=0.02):
    """Identifies non-dominated architectures on (Expressivity, Trainability vs. Param Count).

    A candidate B dominates A only if B is equal/better across all axes within tolerance `tol`
    and strictly superior by at least `tol` in at least one axis. This prevents trivial noise
    or minor topological differences in rank ratio from falsely dominating clean candidates.
    """
    valid = [c for c in candidates if not c.vetoed]
    if len(valid) < 2:
      for c in valid:
        c.is_dominated = False
      return

    for a in valid:
      a.is_dominated = False
      for b in valid:
        if a == b:
          continue
        better_or_equal = (b.metrics["expressivity"] >= a.metrics["expressivity"] - tol and
                           b.metrics["trainability"] >= a.metrics["trainability"] - tol and
                           b.metrics["param_count"] <= a.metrics["param_count"] * (1.0 + tol))
        strictly_better = (b.metrics["expressivity"] > a.metrics["expressivity"] + tol or
                           b.metrics["trainability"] > a.metrics["trainability"] + tol or
                           b.metrics["param_count"] < a.metrics["param_count"] * (1.0 - tol))
        if better_or_equal and strictly_better:
          a.is_dominated = True
          break

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

    # Normalize param counts across candidates
    all_params = [c.metrics.get("param_count", 1.0) for c in candidates]
    min_p = min(all_params) if all_params else 1.0
    max_p = max(all_params) if all_params else 1.0

    for cand in candidates:
      if cand.vetoed:
        cand.composite_score = 0.0
        continue

      m = cand.metrics
      p_val = m.get("param_count", 1.0)
      if max_p > min_p:
        eff_score = 1.0 - ((p_val - min_p) / (max_p - min_p))
      else:
        eff_score = 1.0

      cand.sub_scores = {
        "trainability": m["trainability"],
        "expressivity": m["expressivity"],
        "efficiency": eff_score,
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
        "Pareto-dominated by another candidate with superior expressivity/efficiency trade-off.")
      return

    diag = cand.raw_diagnostics
    warnings = []
    if diag.get("quantization", {}).get("status") == "warning":
      warnings.append("High activation kurtosis; consider adding RMSNorm or smoothing before PTQ.")
    if diag.get("cka", {}).get("status") == "warning":
      warnings.append("Identified redundant consecutive layers; depth can be pruned.")
    if diag.get("curvature", {}).get("status") == "warning":
      warnings.append("Loss surface has high local curvature; recommend lower learning rate.")

    if warnings:
      cand.verdict = "CONDITIONALLY_VIABLE"
      cand.recommendations.extend(warnings)
    else:
      cand.verdict = "PROMOTE"
      cand.recommendations.append(
        "Pareto-optimal candidate with clean diagnostic health. Ready for training.")
