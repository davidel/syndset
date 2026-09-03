"""Synthetic tabular and manifold datasets for dense architecture evaluation."""

import math

import torch

from syndset.domains.base import SyntheticDataset


class ConcentricHyperspheresDataset(SyntheticDataset):
  """Generates nested D-dimensional spherical shells.

  Because linear hyperplanes cannot separate concentric spheres, this dataset
  evaluates non-linear representational capacity and depth efficiency of
  feed-forward neural networks and MLPs.
  """

  def __init__(self, num_samples=1000, num_classes=3, dim=8, noise_std=0.05, seed=42):
    """Initializes concentric hyperspheres.

    Args:
      num_samples: Total number of points to generate (default: 1000).
      num_classes: Number of concentric spherical shells (default: 3).
      dim: Dimensionality of feature space (default: 8).
      noise_std: Standard deviation of radial noise (default: 0.05).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._num_classes = num_classes
    self._dim = dim
    self._noise_std = noise_std

    generator = torch.Generator().manual_seed(seed)
    data_list = []
    labels_list = []

    samples_per_class = num_samples // num_classes
    remainder = num_samples % num_classes

    for class_idx in range(num_classes):
      count = samples_per_class + (1 if class_idx < remainder else 0)
      if count == 0:
        continue

      # Sample random uniform directions on unit hypersphere
      directions = torch.randn(count, dim, generator=generator)
      norm = torch.norm(directions, p=2, dim=1, keepdim=True)
      unit_directions = directions / (norm + 1e-12)

      # Shell radii: 1.0, 2.0, 3.0, ...
      base_radius = float(class_idx + 1)
      radial_noise = torch.randn(count, 1, generator=generator) * noise_std
      radii = (base_radius + radial_noise).clamp(min=0.1)

      points = unit_directions * radii
      targets = torch.full((count,), class_idx, dtype=torch.long)

      data_list.append(points)
      labels_list.append(targets)

    all_data = torch.cat(data_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    # Shuffle dataset
    perm = torch.randperm(num_samples, generator=generator)
    self._data = all_data[perm]
    self._labels = all_labels[perm]

  @property
  def num_classes(self):
    """Returns the number of concentric classes."""
    return self._num_classes

  @property
  def dim(self):
    """Returns the feature dimensionality."""
    return self._dim

  @property
  def noise_std(self):
    """Returns the standard deviation of radial noise."""
    return self._noise_std

  @property
  def data(self):
    """Returns the matrix of generated feature points."""
    return self._data

  @property
  def labels(self):
    """Returns the class labels corresponding to concentric shells."""
    return self._labels

  def __getitem__(self, idx):
    """Returns the feature vector and class label."""
    return self._data[idx], self._labels[idx]

  def description(self):
    """Returns a description of the dataset."""
    return f"Concentric Hyperspheres ({self._num_classes} shells, {self._dim}D)"


class SparseXORDataset(SyntheticDataset):
  """Generates high-dimensional inputs where only a few sparse coordinates determine parity.

  This tests an architecture's capability for coordinate selection and feature
  interaction amidst a sea of irrelevant distractor features.
  """

  def __init__(self, num_samples=1000, total_dim=64, active_dim=3, seed=42):
    """Initializes the sparse XOR dataset.

    Args:
      num_samples: Total number of samples (default: 1000).
      total_dim: Total number of features (default: 64).
      active_dim: Number of informative coordinates (default: 3).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._total_dim = total_dim
    self._active_dim = active_dim

    if active_dim > total_dim:
      raise ValueError("active_dim cannot exceed total_dim.")

    generator = torch.Generator().manual_seed(seed)

    # Distractor coordinates are standard Gaussian noise
    features = torch.randn(num_samples, total_dim, generator=generator)

    # Binary active coordinates {-1, +1}
    active_bits = torch.randint(0, 2, (num_samples, active_dim), generator=generator)
    features[:, :active_dim] = active_bits.float() * 2.0 - 1.0

    # Target parity: sum of active bits mod 2
    targets = active_bits.sum(dim=1) % 2

    self._data = features
    self._labels = targets.long()

  @property
  def total_dim(self):
    """Returns the total number of features."""
    return self._total_dim

  @property
  def active_dim(self):
    """Returns the number of active informative coordinates."""
    return self._active_dim

  @property
  def data(self):
    """Returns the matrix of high-dimensional feature vectors."""
    return self._data

  @property
  def labels(self):
    """Returns the parity target labels."""
    return self._labels

  def __getitem__(self, idx):
    """Returns the high-dimensional feature vector and binary XOR target."""
    return self._data[idx], self._labels[idx]

  def description(self):
    """Returns a description of the dataset."""
    return f"Sparse XOR ({self._active_dim} active of {self._total_dim} total features)"


class IllConditionedRegressionDataset(SyntheticDataset):
  """Generates linear/polynomial regression data with ill-conditioned feature covariance.

  Eigenvalues of the covariance matrix decay exponentially, forcing a high
  condition number. This exposes whether an architecture or optimizer is
  fragile to correlated inputs and gradient curvature.
  """

  def __init__(self, num_samples=1000, dim=32, condition_number=1e4, noise_std=0.01, seed=42):
    """Initializes ill-conditioned regression.

    Args:
      num_samples: Number of samples (default: 1000).
      dim: Number of features (default: 32).
      condition_number: Ratio of largest to smallest singular value (default: 1e4).
      noise_std: Additive Gaussian target noise (default: 0.01).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._dim = dim
    self._condition_number = condition_number

    generator = torch.Generator().manual_seed(seed)

    # Construct singular values decaying geometrically
    log_decay = torch.linspace(0.0, -math.log10(condition_number), dim)
    singular_values = 10.0**log_decay

    # Random orthogonal projection
    rand_matrix = torch.randn(dim, dim, generator=generator)
    q_proj, _ = torch.linalg.qr(rand_matrix)
    cov_half = q_proj @ torch.diag(singular_values)

    # Generate correlated input features
    white_noise = torch.randn(num_samples, dim, generator=generator)
    x = white_noise @ cov_half.T

    # Generate targets with a fixed weight vector
    true_weights = torch.randn(dim, 1, generator=generator)
    y = x @ true_weights + torch.randn(num_samples, 1, generator=generator) * noise_std

    self._data = x
    self._targets = y.squeeze(-1)

  @property
  def dim(self):
    """Returns the number of input features."""
    return self._dim

  @property
  def condition_number(self):
    """Returns the matrix condition number."""
    return self._condition_number

  @property
  def data(self):
    """Returns the ill-conditioned input features."""
    return self._data

  @property
  def targets(self):
    """Returns the continuous regression targets."""
    return self._targets

  def __getitem__(self, idx):
    """Returns the ill-conditioned input vector and regression target scalar."""
    return self._data[idx], self._targets[idx]

  def description(self):
    """Returns a description of the dataset."""
    return f"Ill-Conditioned Regression (dim={self._dim}, cond={self._condition_number:.1e})"


class TwoSpiralsDataset(SyntheticDataset):
  """Generates two interlocking Archimedean spirals.

  The two-spirals problem (Lang & Witbrock, 1988) is the canonical benchmark
  for non-linear classification and topological manifold untangling.
  Because the two spirals wind around each other with continuous phase rotation,
  they cannot be separated by linear coordinate transforms or shallow networks
  without sufficient depth and non-linear capacity.
  """

  def __init__(self, num_samples=1000, turns=2.5, noise_std=0.05, seed=42):
    """Initializes the two spirals dataset.

    Args:
      num_samples: Total number of 2D coordinates to generate (default: 1000).
      turns: Number of revolutions for each spiral (default: 2.5).
      noise_std: Standard deviation of Cartesian coordinate noise (default: 0.05).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._turns = turns
    self._noise_std = noise_std

    generator = torch.Generator().manual_seed(seed)
    n_half = num_samples // 2
    n_other = num_samples - n_half

    # Spiral 1 (label 0)
    theta1 = torch.linspace(0.0, 2.0 * math.pi * turns, n_half)
    r1 = theta1 / (2.0 * math.pi * turns)
    noise_x1 = torch.randn(n_half, generator=generator) * noise_std
    noise_y1 = torch.randn(n_half, generator=generator) * noise_std
    x1 = r1 * torch.cos(theta1) + noise_x1
    y1 = r1 * torch.sin(theta1) + noise_y1
    pts1 = torch.stack([x1, y1], dim=1)
    lbl1 = torch.zeros(n_half, dtype=torch.long)

    # Spiral 2 (label 1, rotated by pi radians)
    theta2 = torch.linspace(0.0, 2.0 * math.pi * turns, n_other)
    r2 = theta2 / (2.0 * math.pi * turns)
    noise_x2 = torch.randn(n_other, generator=generator) * noise_std
    noise_y2 = torch.randn(n_other, generator=generator) * noise_std
    x2 = -r2 * torch.cos(theta2) + noise_x2
    y2 = -r2 * torch.sin(theta2) + noise_y2
    pts2 = torch.stack([x2, y2], dim=1)
    lbl2 = torch.ones(n_other, dtype=torch.long)

    all_data = torch.cat([pts1, pts2], dim=0)
    all_labels = torch.cat([lbl1, lbl2], dim=0)

    perm = torch.randperm(num_samples, generator=generator)
    self._data = all_data[perm]
    self._labels = all_labels[perm]

  @property
  def turns(self):
    """Returns the number of spiral revolutions."""
    return self._turns

  @property
  def noise_std(self):
    """Returns the noise standard deviation."""
    return self._noise_std

  @property
  def data(self):
    """Returns the matrix of 2D spiral coordinates."""
    return self._data

  @property
  def labels(self):
    """Returns the binary spiral identity labels."""
    return self._labels

  def __getitem__(self, idx):
    """Returns the 2D coordinate vector and spiral class label."""
    return self._data[idx], self._labels[idx]

  def description(self):
    """Returns a description of the dataset."""
    return f"Two Spirals ({self._turns} turns, noise={self._noise_std})"
