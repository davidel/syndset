"""Pytest fixtures and toy dummy neural architectures for test suite."""

import pytest
import torch.nn as nn


class ToyMLP(nn.Module):
  """Simple Multi-Layer Perceptron for tabular sanity testing."""

  def __init__(self, in_features=16, hidden_dim=32, out_features=3):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_features)
    )

  def forward(self, x):
    return self.net(x)


class ToyConvNet(nn.Module):
  """Simple 2D Convolutional network for vision testing."""

  def __init__(self, in_channels=1, num_classes=4):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, 8, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
    )
    self.classifier = nn.Linear(8 * 4 * 4, num_classes)

  def forward(self, x):
    h = self.conv(x)
    h = h.flatten(start_dim=1)
    return self.classifier(h)


class ToyTransformer(nn.Module):
  """Minimal Transformer encoder model for sequence testing."""

  def __init__(self, vocab_size=64, d_model=32, num_heads=2):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, d_model)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, nhead=num_heads, dim_feedforward=64, batch_first=True
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
    self.head = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    h = self.embed(x)
    out = self.encoder(h)
    return self.head(out[:, -1, :])


class DisconnectedParamModel(nn.Module):
  """Model with an intentionally unused parameter branch to test zero-grad detection."""

  def __init__(self, in_features=8, out_features=2):
    super().__init__()
    self.used_layer = nn.Linear(in_features, out_features)
    self.unused_layer = nn.Linear(in_features, out_features)

  def forward(self, x):
    return self.used_layer(x)


@pytest.fixture
def toy_mlp():
  return ToyMLP()


@pytest.fixture
def toy_convnet():
  return ToyConvNet()


@pytest.fixture
def toy_transformer():
  return ToyTransformer()


@pytest.fixture
def disconnected_model():
  return DisconnectedParamModel()
