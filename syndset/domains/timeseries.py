"""Synthetic time series datasets for spectral bias and temporal lag evaluation."""

import math

import torch

from syndset.domains.base import SyntheticDataset


class HarmonicSuperpositionDataset(SyntheticDataset):
  """Generates multi-frequency harmonic time-series with controllable spectral modes.

  Neural networks often exhibit spectral bias (learning low frequencies first).
  This dataset superimposes multiple frequency bands to evaluate whether an
  architecture can resolve high-frequency signals or dampens them out.

  What this dataset generates:
    1. Temporal Time Mesh:
       - Evaluates time steps t = linspace(0.0, 1.0, seq_len).
    2. Multi-Harmonic Superposition:
       - Superimposes `num_harmonics` sinusoidal waves with harmonic frequencies and decaying power:
         For harmonic h in 1..num_harmonics:
           frequency f_h = 2 * h
           amplitude a_h = 1.0 / sqrt(h)
           phase phi_h ~ Uniform(0, 2*pi)
           signal(t) = sum_{h=1}^H a_h * sin(2*pi*f_h*t + phi_h).
    3. Noise Injection & 1-Step Forecast Split:
       - Adds Gaussian noise: noisy_signal = signal + N(0, noise_std^2).
       - Inputs: past window `noisy_signal[:-1]` (length: seq_len - 1).
       - Targets: clean 1-step shifted future `signal[1:]` (length: seq_len - 1).

  Tensor Layout:
    inputs[idx]:  2D tensor of shape `(seq_len - 1, 1)` with noisy past values (float32)
    targets[idx]: 2D tensor of shape `(seq_len - 1, 1)` with clean 1-step forecast (float32)

  Concrete Example:
    Suppose seq_len=8, num_harmonics=2:
      True underlying multi-frequency wave generated across 8 time steps
      inputs[0]:  7 time steps of noisy signal [x_0, x_1, x_2, x_3, x_4, x_5, x_6]
      targets[0]: 7 time steps of clean future [s_1, s_2, s_3, s_4, s_5, s_6, s_7]

  Attributes:
    inputs: Tensor of shape `(num_samples, seq_len - 1, 1)` with dtype `torch.float32`.
    targets: Tensor of shape `(num_samples, seq_len - 1, 1)` with dtype `torch.float32`.

  Item Format:
    Indexing `dataset[idx]` returns `(input_seq, target_seq)` where:
      - `input_seq`: 2D `torch.float32` tensor of shape `(seq_len - 1, 1)`.
      - `target_seq`: 2D `torch.float32` tensor of shape `(seq_len - 1, 1)`
        (1-step shifted forecast).
  """

  def __init__(self, num_samples=1000, seq_len=64, num_harmonics=4, noise_std=0.05, seed=42):
    """Initializes harmonic superposition time-series.

    Args:
      num_samples: Total number of time series sequences (default: 1000).
      seq_len: Number of time steps per sequence (default: 64).
        Input sequence length is `seq_len - 1`.
      num_harmonics: Number of sinusoidal frequency components (default: 4).
      noise_std: Additive Gaussian noise standard deviation (default: 0.05).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._seq_len = seq_len
    self._num_harmonics = num_harmonics
    self._noise_std = noise_std

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
  def seq_len(self):
    """Returns the sequence length."""
    return self._seq_len

  @property
  def num_harmonics(self):
    """Returns the number of harmonic components."""
    return self._num_harmonics

  @property
  def noise_std(self):
    """Returns the noise standard deviation."""
    return self._noise_std

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
    return f"Harmonic Superposition ({self._num_harmonics} harmonics, len={self._seq_len})"


class AutoregressiveLagDataset(SyntheticDataset):
  """Generates non-linear time series with explicit long-range lag dependencies.

  Evaluates whether recurrent, convolutional (TCN), or state-space models
  can preserve and attend to distant historical time steps without decay.

  What this dataset generates:
    1. Prefix Initialization:
       - Initializes a series of length `total_steps`.
       - Seeds the initial prefix [0 : max_lag] with random normal values ~ N(0, 0.25).
    2. Non-Linear Delayed Lag Dynamics:
       - For t = max_lag, ..., total_steps - 1:
         Computes linear combination of specified past lag points:
         val = sum_l w_l * series[t - lag_l] (alternating weights +0.5, -0.4).
         Applies non-linear tanh activation and noise:
         series[t] = tanh(val) + N(0, noise_std^2).
    3. Input Sequence & Autoregressive Next-Step Target:
       - `sequences`: past sequence `series[:-1]` of length `total_steps - 1`.
       - `targets`: 1-step shifted sequence `series[1:]` of length `total_steps - 1`.

  Tensor Layout:
    sequences[idx]: 2D tensor of shape `(total_steps - 1, 1)` (float32)
    targets[idx]:   2D tensor of shape `(total_steps - 1, 1)` (float32)

  Concrete Example:
    Suppose total_steps=32, lags=(5, 20):
      Step t=20 explicitly depends on step t=15 (lag 5) and step t=0 (lag 20).
      Tests if the architecture can bridge a 20-step temporal gap to accurately
      predict future transitions.

  Attributes:
    sequences: Tensor of shape `(num_samples, total_steps - 1, 1)` with dtype `torch.float32`.
    targets: Tensor of shape `(num_samples, total_steps - 1, 1)` with dtype `torch.float32`.

  Item Format:
    Indexing `dataset[idx]` returns `(input_seq, target_seq)` where:
      - `input_seq`: 2D `torch.float32` tensor of shape `(total_steps - 1, 1)`.
      - `target_seq`: 2D `torch.float32` tensor of shape `(total_steps - 1, 1)`.
  """

  def __init__(self, num_samples=1000, total_steps=64, lags=(5, 20), noise_std=0.02, seed=42):
    """Initializes autoregressive lag time series.

    Args:
      num_samples: Total sequences to generate (default: 1000).
      total_steps: Sequence length (default: 64). Input sequence length is `total_steps - 1`.
      lags: Tuple of integer lag indices that causally govern the series.
      noise_std: Additive noise level (default: 0.02).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._total_steps = total_steps
    self._lags = lags
    self._noise_std = noise_std

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
  def total_steps(self):
    """Returns the total sequence length."""
    return self._total_steps

  @property
  def lags(self):
    """Returns the governing lag indices."""
    return self._lags

  @property
  def noise_std(self):
    """Returns the noise standard deviation."""
    return self._noise_std

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
    return f"Autoregressive Lags {self._lags} (len={self._total_steps})"
