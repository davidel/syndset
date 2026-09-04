"""Reporting and scorecard formatting for architecture audit results."""


class AuditReport:
  """Aggregates and formats diagnostic check findings into a readable scorecard.

  Provides clear console summaries with actionable insights when a check fails
  or produces warnings.
  """

  def __init__(self, model_name="Model"):
    """Initializes an empty audit report.

    Args:
      model_name: Optional string identifier for the inspected architecture.
    """
    self._model_name = model_name
    self._checks = []
    self._overall_status = "passed"

  @property
  def model_name(self):
    """Returns the identifier of the inspected architecture."""
    return self._model_name

  @property
  def checks(self):
    """Returns a list copy of all recorded diagnostic check records."""
    return list(self._checks)

  @property
  def overall_status(self):
    """Returns the aggregate health status ('passed', 'warning', or 'failed')."""
    return self._overall_status

  def add_check(self, name, status, summary, details=None):
    """Appends an individual diagnostic check result to the report.

    Args:
      name: Name of the diagnostic probe (e.g. 'Gradient Flow').
      status: String status ('healthy', 'passed', 'warning', or 'failed').
      summary: Short one-line takeaway.
      details: Optional list of detailed notes or warnings.
    """
    normalized_status = status.lower()
    if normalized_status in ("healthy", "passed"):
      tag = "PASS"
    elif normalized_status in ("warning", "warn", "slow"):
      tag = "WARN"
      if self._overall_status != "failed":
        self._overall_status = "warning"
    else:
      tag = "FAIL"
      self._overall_status = "failed"

    self._checks.append({
      "name": name,
      "tag": tag,
      "summary": summary,
      "details": details or [],
    })

  def is_healthy(self):
    """Returns True if no failures were recorded in any check."""
    return self._overall_status != "failed"

  def to_dict(self):
    """Returns a structured dictionary representation of the report."""
    return {
      "model_name": self._model_name,
      "overall_status": self._overall_status,
      "checks": list(self._checks),
    }

  def summary(self):
    """Renders the scorecard into a formatted string table.

    Returns:
      A multi-line formatted string.
    """
    lines = []
    separator = "=" * 70
    lines.append(separator)
    lines.append(f"  SYNDSET ARCHITECTURE AUDIT REPORT: {self._model_name}")
    lines.append(f"  Overall Health: [{self._overall_status.upper()}]")
    lines.append(separator)
    lines.append(f"{'Check':<25} | {'Status':<8} | {'Finding'}")
    lines.append("-" * 70)

    for check in self._checks:
      tag_str = f"[{check['tag']}]"
      lines.append(f"{check['name']:<25} | {tag_str:<8} | {check['summary']}")
      for detail in check["details"]:
        lines.append(f"  -> {detail}")

    lines.append(separator)
    return "\n".join(lines)

  def print_summary(self):
    """Prints the formatted scorecard directly to standard output."""
    print(self.summary())
