"""Synthetic time series datasets for spectral bias and temporal lag evaluation."""

import math

import torch

from syndset.domains.base import SyntheticDataset


class HarmonicSuperpositionDataset(SyntheticDataset):
  """Generates multi-frequency harmonic time-series with controllable spectral modes.

  Neural networks often exhibit spectral bias (learning low frequencies first).
  This dataset superimposes multiple frequency bands to evaluate whether an
  architecture can resolve high-frequency signals or dampens them out.
  """

  def __init__(self, num_samples=1000, seq_len=64, num_harmonics=4, noise_std=0.05, seed=42):
    """Initializes harmonic superposition time-series.

    Args:
      num_samples: Total number of time series sequences (default: 1000).
      seq_len: Number of time steps per sequence (default: 64).
      num_harmonics: Number of sinusoidal frequency components (default: 4).
      noise_std: Additive Gaussian noise standard deviation (default: 0.05).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self.seq_len = seq_len
    self.num_harmonics = num_harmonics
    self.noise_std = noise_std

    generator = torch.Generator().manual_seed(seed)
    time_steps = torch.linspace(0.0, 1.0, seq_len)

    inputs_list = []
    targets_list = []

    for _ in range(num_samples):
      signal = torch.zeros(seq_len)

      for h in range(1, num_harmonics + 1):
        freq = float(h * 2)
        amplitude = 1.0 / math.sqrt(h)
        phase = torch.rand(1, generator=generator).item() * 2.0 * math.pi
        signal += amplitude * torch.sin(2.0 * math.pi * freq * time_steps + phase)

      noise = torch.randn(seq_len, generator=generator) * noise_std
      noisy_signal = signal + noise

      # Forecast task: past values predict next step, or signal reconstruction
      inputs_list.append(noisy_signal[:-1].unsqueeze(-1))
      targets_list.append(signal[1:].unsqueeze(-1))

    self._inputs = torch.stack(inputs_list, dim=0)
    self._targets = torch.stack(targets_list, dim=0)

  @property
  def inputs(self):
    """Returns the generated time-series input windows."""
    return self._inputs

  @property
  def targets(self):
    """Returns the forecasted target series."""
    return self._targets

  def __getitem__(self, idx):
    """Returns the input time-series window and forecast target sequence."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the dataset."""
    return f"Harmonic Superposition ({self.num_harmonics} harmonics, len={self.seq_len})"


class AutoregressiveLagDataset(SyntheticDataset):
  """Generates non-linear time series with explicit long-range lag dependencies.

  Evaluates whether recurrent, convolutional (TCN), or state-space models
  can preserve and attend to distant historical time steps without decay.
  """

  def __init__(self, num_samples=1000, total_steps=64, lags=(5, 20), noise_std=0.02, seed=42):
    """Initializes autoregressive lag time series.

    Args:
      num_samples: Total sequences to generate (default: 1000).
      total_steps: Sequence length (default: 64).
      lags: Tuple of integer lag indices that causally govern the series.
      noise_std: Additive noise level (default: 0.02).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self.total_steps = total_steps
    self.lags = lags
    self.noise_std = noise_std

    max_lag = max(lags)
    if total_steps <= max_lag:
      raise ValueError(f"total_steps ({total_steps}) must be greater than max lag ({max_lag}).")

    generator = torch.Generator().manual_seed(seed)
    sequences_list = []
    targets_list = []

    for _ in range(num_samples):
      series = torch.zeros(total_steps)
      # Seed initial steps with random values
      series[:max_lag] = torch.randn(max_lag, generator=generator) * 0.5

      for t in range(max_lag, total_steps):
        # Non-linear interaction between specified lags
        val = 0.0
        for idx, lag in enumerate(lags):
          weight = 0.5 if idx % 2 == 0 else -0.4
          val += weight * series[t - lag]
        val = math.tanh(val) + (torch.randn(1, generator=generator).item() * noise_std)
        series[t] = val

      sequences_list.append(series[:-1].unsqueeze(-1))
      targets_list.append(series[1:].unsqueeze(-1))

    self._sequences = torch.stack(sequences_list, dim=0)
    self._targets = torch.stack(targets_list, dim=0)

  @property
  def sequences(self):
    """Returns the input autoregressive sequences."""
    return self._sequences

  @property
  def targets(self):
    """Returns the shifted autoregressive target sequences."""
    return self._targets

  def __getitem__(self, idx):
    """Returns sequence input and shifted autoregressive targets."""
    return self._sequences[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Autoregressive Lags {self.lags} (len={self.total_steps})"
