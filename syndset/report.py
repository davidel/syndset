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
    self.model_name = model_name
    self.checks = []
    self.overall_status = "passed"

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
      if self.overall_status != "failed":
        self.overall_status = "warning"
    else:
      tag = "FAIL"
      self.overall_status = "failed"

    self.checks.append(
      {
        "name": name,
        "tag": tag,
        "summary": summary,
        "details": details or [],
      }
    )

  def is_healthy(self):
    """Returns True if no failures were recorded in any check."""
    return self.overall_status != "failed"

  def to_dict(self):
    """Returns a structured dictionary representation of the report."""
    return {
      "model_name": self.model_name,
      "overall_status": self.overall_status,
      "checks": self.checks,
    }

  def summary(self):
    """Renders the scorecard into a formatted string table.

    Returns:
      A multi-line formatted string.
    """
    lines = []
    separator = "=" * 70
    lines.append(separator)
    lines.append(f"  SYNDSET ARCHITECTURE AUDIT REPORT: {self.model_name}")
    lines.append(f"  Overall Health: [{self.overall_status.upper()}]")
    lines.append(separator)
    lines.append(f"{'Check':<25} | {'Status':<8} | {'Finding'}")
    lines.append("-" * 70)

    for check in self.checks:
      tag_str = f"[{check['tag']}]"
      lines.append(f"{check['name']:<25} | {tag_str:<8} | {check['summary']}")
      for detail in check["details"]:
        lines.append(f"  -> {detail}")

    lines.append(separator)
    return "\n".join(lines)

  def print_summary(self):
    """Prints the formatted scorecard directly to standard output."""
    print(self.summary())
