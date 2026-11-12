"""Tests for tournament orchestration across model candidates."""

import torch
import torch.nn as nn

from syndset.tournament import (
  Tournament,
  tournament_llm,
  tournament_tabular,
  tournament_vision,
)


def test_tournament_general():
  """Tests general tournament execution on multiple MLP architectures."""
  torch.manual_seed(42)
  model_a = nn.Sequential(
    nn.Linear(8, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
  )
  model_b = nn.Sequential(
    nn.Linear(8, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
  )

  inputs = torch.randn(8, 8)
  targets = torch.randint(0, 2, (8,))

  tourney = Tournament(domain="general", profile="balanced")
  report = tourney.evaluate(
    {
      "ModelA": model_a,
      "ModelB": model_b
    },
    inputs,
    targets,
  )

  assert len(report.candidates) == 2
  assert report.domain == "general"
  summary_str = report.summary()
  assert "SYNDSET ARCHITECTURE TOURNAMENT" in summary_str
  assert "ModelA" in summary_str
  assert "ModelB" in summary_str


def test_tournament_tabular():
  """Tests tournament_tabular helper."""
  model_1 = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3))
  model_2 = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))

  report = tournament_tabular({"M1": model_1, "M2": model_2}, dim=16, num_classes=3, batch_size=8)
  assert len(report.candidates) == 2
  data = report.to_dict()
  assert data["domain"] == "tabular"
  assert len(data["rankings"]) == 2


def test_tournament_vision():
  """Tests tournament_vision helper."""
  model_1 = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(4, 8),
  )
  report = tournament_vision({"V1": model_1}, img_size=16, channels=1, num_classes=8, batch_size=4)
  assert len(report.candidates) == 1
  assert report.domain == "vision"


def test_tournament_llm():
  """Tests tournament_llm helper."""

  class ToySeq(nn.Module):

    def __init__(self):
      super().__init__()
      self.emb = nn.Embedding(32, 16)
      self.fc = nn.Linear(16, 32)

    def forward(self, x):
      return self.fc(self.emb(x)[:, -1, :])

  model = ToySeq()
  report = tournament_llm({"LLM1": model}, num_pairs=4, vocab_size=32, batch_size=4)
  assert len(report.candidates) == 1
  assert report.domain == "llm"
