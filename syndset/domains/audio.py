"""Synthetic audio waveforms, FM chirps, and sound event detection datasets."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


@register_dataset("audio/fm_chirp")
@register_dataset("audio/chirp")
class FMChirpDataset(SyntheticDataset):
  """Generates 1D frequency-modulated (FM) chirps and multi-harmonic signals.

  Tests 1D temporal convolutions, spectrogram/STFT autograd layers,
  and frequency resolution in raw waveform neural backbones.
  """

  def __init__(
    self,
    num_samples: int = 500,
    seq_len: int = 256,
    sample_rate: float = 1000.0,
    noise_std: float = 0.02,
    seed: int = 42,
  ):
    """Initializes FM chirp audio dataset.

    Args:
      num_samples: Number of 1D audio signals to generate (default: 500).
      seq_len: Number of discrete temporal samples per signal (default: 256).
      sample_rate: Sampling frequency in Hertz (default: 1000.0).
      noise_std: Additive Gaussian noise level (default: 0.02).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._seq_len = seq_len
    self._sample_rate = sample_rate
    self._noise_std = noise_std

    generator = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.0, seq_len / sample_rate, seq_len)

    self._waveforms = []
    self._initial_freqs = []
    self._chirp_rates = []

    for _ in range(num_samples):
      f0 = torch.rand(1, generator=generator).item() * 40.0 + 10.0  # [10, 50] Hz
      beta = torch.rand(1, generator=generator).item() * 100.0 + 20.0  # chirp rate
      phi = torch.rand(1, generator=generator).item() * 2.0 * math.pi

      phase = 2.0 * math.pi * (f0 * t + 0.5 * beta * (t**2)) + phi
      signal = torch.sin(phase)

      noise = torch.randn(seq_len, generator=generator) * noise_std
      noisy_signal = (signal + noise).unsqueeze(0)  # (1, T)

      self._waveforms.append(noisy_signal)
      self._initial_freqs.append(f0)
      self._chirp_rates.append(beta)

    self._waveforms_tensor = torch.stack(self._waveforms, dim=0)
    self._initial_freqs_tensor = torch.tensor(self._initial_freqs, dtype=torch.float32)
    self._chirp_rates_tensor = torch.tensor(self._chirp_rates, dtype=torch.float32)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing 1D waveform, f0, and chirp rate."""
    return DataSample(
      waveform=self._waveforms_tensor[idx],
      initial_freq=self._initial_freqs_tensor[idx],
      chirp_rate=self._chirp_rates_tensor[idx],
    )

  def description(self) -> str:
    """Returns a description of the dataset."""
    return f"FM Chirps (1x{self._seq_len} samples @ {self._sample_rate}Hz)"


@register_dataset("audio/sound_events")
class PolyphonicSoundEventDataset(SyntheticDataset):
  """Procedurally generates polyphonic audio signals with exact event timestamps.

  Evaluates sound event localization, onset/offset detection, and multi-label
  temporal audio tagging.
  """

  def __init__(
    self,
    num_samples: int = 500,
    seq_len: int = 512,
    num_classes: int = 4,
    seed: int = 42,
  ):
    """Initializes polyphonic sound event dataset.

    Args:
      num_samples: Total number of audio sequences (default: 500).
      seq_len: Length of the discrete 1D waveform (default: 512).
      num_classes: Number of distinct frequency event classes (default: 4).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._seq_len = seq_len
    self._num_classes = num_classes

    generator = torch.Generator().manual_seed(seed)
    time_steps = torch.linspace(0.0, 1.0, seq_len)

    self._waveforms = []
    self._event_masks = []

    # Frequency carriers for classes
    carrier_freqs = [20.0 * (k + 1) for k in range(num_classes)]

    for _ in range(num_samples):
      signal = torch.zeros(seq_len, dtype=torch.float32)
      mask = torch.zeros((num_classes, seq_len), dtype=torch.float32)

      for c in range(num_classes):
        # 50% chance of event occurrence
        if torch.rand(1, generator=generator).item() > 0.5:
          t_start = torch.rand(1, generator=generator).item() * 0.6
          dur = torch.rand(1, generator=generator).item() * 0.3 + 0.1
          t_end = min(1.0, t_start + dur)

          active = (time_steps >= t_start) & (time_steps <= t_end)
          freq = carrier_freqs[c]
          tone = torch.sin(2.0 * math.pi * freq * time_steps) * active.float()
          signal += tone
          mask[c] = active.float()

      self._waveforms.append(signal.unsqueeze(0))
      self._event_masks.append(mask)

    self._waveforms_tensor = torch.stack(self._waveforms, dim=0)
    self._event_masks_tensor = torch.stack(self._event_masks, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing 1D waveform and per-class active time masks."""
    return DataSample(
      waveform=self._waveforms_tensor[idx],
      event_mask=self._event_masks_tensor[idx],
    )

  def description(self) -> str:
    """Returns a description of the sound event dataset."""
    return f"Polyphonic Sound Events ({self._num_classes} classes, 1x{self._seq_len})"
