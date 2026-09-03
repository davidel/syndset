"""End-to-end tests for the audit runner and scorecard generator."""

import torch
import torch.nn as nn

from syndset.audit import (
  audit,
  audit_llm,
  audit_tabular,
  audit_timeseries,
  audit_vision,
)


class ToyRNN(nn.Module):
  """Simple GRU with output projection for time series audit testing."""

  def __init__(self):
    super().__init__()
    self.rnn = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
    self.fc = nn.Linear(16, 1)

  def forward(self, x):
    out, _ = self.rnn(x)
    return self.fc(out)


def test_audit_end_to_end_mlp(toy_mlp):
  """Tests running a full audit battery on a tabular MLP."""
  inputs = torch.randn(8, 16)
  targets = torch.randint(0, 3, (8,))

  report = audit(toy_mlp, inputs, sample_target=targets, model_name="TestMLP")

  assert report.is_healthy()
  summary_text = report.summary()
  assert "TestMLP" in summary_text
  assert "Gradient Flow" in summary_text
  assert "Effective Rank" in summary_text
  assert "Overfit Capacity" in summary_text

  report_dict = report.to_dict()
  assert "checks" in report_dict
  assert len(report_dict["checks"]) >= 3


def test_audit_domain_shortcuts(toy_mlp, toy_convnet, toy_transformer):
  """Tests the convenience audit functions for each neural network domain."""
  # 1. Tabular shortcut
  tabular_report = audit_tabular(toy_mlp, dim=16, num_classes=3, batch_size=8)
  assert tabular_report is not None

  # 2. Vision shortcut
  vision_report = audit_vision(toy_convnet, img_size=32, channels=1, batch_size=4)
  assert vision_report is not None

  # 3. LLM shortcut
  llm_report = audit_llm(toy_transformer, num_pairs=4, vocab_size=64, batch_size=4)
  assert llm_report is not None

  # 4. Time series shortcut
  rnn_model = ToyRNN()
  ts_report = audit_timeseries(rnn_model, seq_len=16, batch_size=4)
  assert ts_report is not None
