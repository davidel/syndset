"""Synthetic tabular and manifold datasets for dense architecture evaluation."""

import math

import torch

from syndset.domains.base import SyntheticDataset


class ConcentricHyperspheresDataset(SyntheticDataset):
  """Generates nested D-dimensional spherical shells.

  Because linear hyperplanes cannot separate concentric spheres, this dataset
  evaluates non-linear representational capacity and depth efficiency of
  feed-forward neural networks and MLPs.

  What this dataset generates:
    1. Spherical Shell Manifolds:
       - Generates `num_samples` points partitioned evenly across `num_classes` concentric shells.
    2. Direction Sampling:
       - Samples isotropic uniform directions on the unit hypersphere in R^dim by normalizing
         Gaussian vectors: u = z / (||z||_2 + 1e-12), where z ~ N(0, I_dim).
    3. Radial Shell Scaling:
       - Sets base radius for shell class c as `c + 1` (e.g. 1.0, 2.0, 3.0, ...).
       - Adds radial Gaussian perturbation N(0, noise_std^2) and clamps min radius to 0.1.
       - Multiplies unit direction u by radius r to yield Cartesian coordinate vector x = u * r.
    4. Target Output:
       - Target label is the integer shell index c in [0, num_classes - 1].

  Tensor Layout:
    data[idx]:   1D tensor of shape `(dim,)` (float32)
    labels[idx]: scalar tensor with shell class index in `[0, num_classes - 1]` (int64)

  Concrete Example:
    Suppose dim=4, num_classes=3, noise_std=0.02:
      Shell 0 point (radius ~1.0): [ 0.52, -0.61,  0.33, -0.49 ] -> label: 0
      Shell 1 point (radius ~2.0): [-1.15,  0.88,  1.20, -0.42 ] -> label: 1
      Shell 2 point (radius ~3.0): [ 1.80,  1.55, -1.22,  1.05 ] -> label: 2

  Attributes:
    data: Tensor of shape `(num_samples, dim)` with dtype `torch.float32`.
    labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `[0, num_classes - 1]`.

  Item Format:
    Indexing `dataset[idx]` returns `(features, label)` where:
      - `features`: 1D `torch.float32` tensor of shape `(dim,)`.
      - `label`: scalar `torch.long` tensor with shell class index in `[0, num_classes - 1]`.
  """

  def __init__(self, num_samples=1000, num_classes=3, dim=8, noise_std=0.05, seed=42):
    """Initializes concentric hyperspheres.

    Args:
      num_samples: Total number of points to generate (default: 1000).
      num_classes: Number of concentric spherical shells (default: 3).
      dim: Dimensionality of feature space (default: 8).
      noise_std: Standard deviation of radial noise (default: 0.05).
      seed: Random seed for reproducible generation (default: 42).
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

  What this dataset generates:
    1. Informative Signal Dimensions:
       - Samples `active_dim` binary bits uniformly from {0, 1}.
       - Maps bits to {-1.0, +1.0} and places them in the first `active_dim` feature coordinates.
    2. Irrelevant Distractor Features:
       - Fills the remaining `total_dim - active_dim` coordinates with uninformative Gaussian
         noise N(0, 1).
    3. Parity Target:
       - Target label is the XOR parity of the active bits: targets = sum(active_bits) % 2.
       - Evaluates whether dense layers can select the relevant subspace and form multiplicative
         interactions without overfitting to the high-dimensional noise floor.

  Tensor Layout:
    data[idx]:   1D tensor of shape `(total_dim,)`
                 (first `active_dim` in {-1.0, +1.0}, rest N(0, 1))
    labels[idx]: scalar binary parity label in {0, 1} (int64)

  Concrete Example:
    Suppose total_dim=6, active_dim=3:
      Active bits: [1, 0, 1] -> coordinates: [+1.0, -1.0, +1.0]
      Distractor noise coordinates: [-0.43, 1.25, 0.08]
      Feature vector: [ +1.0, -1.0, +1.0, -0.43,  1.25,  0.08 ]
      Target parity: (1 + 0 + 1) % 2 = 0

  Attributes:
    data: Tensor of shape `(num_samples, total_dim)` with dtype `torch.float32`.
    labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `{0, 1}`.

  Item Format:
    Indexing `dataset[idx]` returns `(features, label)` where:
      - `features`: 1D `torch.float32` tensor of shape `(total_dim,)`.
      - `label`: scalar `torch.long` tensor with binary XOR parity target in `{0, 1}`.
  """

  def __init__(self, num_samples=1000, total_dim=64, active_dim=3, seed=42):
    """Initializes the sparse XOR dataset.

    Args:
      num_samples: Total number of samples (default: 1000).
      total_dim: Total number of features (default: 64).
      active_dim: Number of informative coordinates (default: 3).
      seed: Random seed for reproducible generation (default: 42).
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

  What this dataset generates:
    1. Geometrically Decaying Singular Values:
       - Constructs singular values sigma_i decaying log-linearly from 1.0 down to
         1 / condition_number across `dim` coordinates:
         log_decay = linspace(0.0, -log10(condition_number), dim), sigma_i = 10^log_decay.
    2. Random Orthogonal Transformation:
       - Projects the singular spectrum through a random orthogonal basis Q (from QR factorization).
       - Forms covariance matrix square-root factor cov_half = Q * diag(sigma).
    3. Input Features:
       - Samples white Gaussian noise Z ~ N(0, I) and correlates features via X = Z * cov_half.T.
       - Feature covariance matrix has condition number kappa = sigma_max / sigma_min.
    4. Regression Targets:
       - Computes linear target y = X * w + noise, where w ~ N(0, I) and noise ~ N(0, noise_std^2).

  Tensor Layout:
    data[idx]:    1D tensor of shape `(dim,)` with ill-conditioned covariance (float32)
    targets[idx]: scalar continuous regression target (float32)

  Concrete Example:
    Suppose dim=4, condition_number=1e4:
      Singular values: [1.0, 0.0464, 0.00215, 0.0001]
      Condition number kappa = 1.0 / 0.0001 = 10,000
      Gradient updates along the smallest eigenvalue axis will be 10,000x slower
      than along the principal axis unless adaptive preconditioning is used.

  Attributes:
    data: Tensor of shape `(num_samples, dim)` with dtype `torch.float32`.
    targets: Tensor of shape `(num_samples,)` with dtype `torch.float32`.

  Item Format:
    Indexing `dataset[idx]` returns `(features, target)` where:
      - `features`: 1D `torch.float32` tensor of shape `(dim,)`.
      - `target`: scalar `torch.float32` regression target.
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

  What this dataset generates:
    1. Archimedean Spiral Trajectory:
       - Parameterizes radius as a function of angle: r(theta) = theta / (2 * pi * turns)
         for theta in [0, 2 * pi * turns].
    2. Two Interleaved Branches:
       - Spiral 1 (label 0):
         x1 =  r(theta) * cos(theta) + N(0, noise_std^2)
         y1 =  r(theta) * sin(theta) + N(0, noise_std^2)
       - Spiral 2 (label 1, rotated by 180 degrees / pi radians):
         x2 = -r(theta) * cos(theta) + N(0, noise_std^2)
         y2 = -r(theta) * sin(theta) + N(0, noise_std^2)
    3. Non-Linear Separability:
       - The two branches wind around each other `turns` times without intersecting,
         demanding deep non-linear decision boundary bending.
    4. Output:
       - 2D coordinate vector in R^2 and binary class label in {0, 1}.

  Tensor Layout:
    data[idx]:   1D tensor of shape `(2,)` representing (x, y) coordinates (float32)
    labels[idx]: scalar binary branch label in {0, 1} (int64)

  Concrete Example:
    Suppose turns=2.5, noise_std=0.05:
      Branch 0 coordinate: [ 0.28, -0.74 ] -> label: 0
      Branch 1 coordinate: [-0.31,  0.71 ] -> label: 1

  Attributes:
    data: Tensor of shape `(num_samples, 2)` with dtype `torch.float32`.
    labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `{0, 1}`.

  Item Format:
    Indexing `dataset[idx]` returns `(coords_2d, label)` where:
      - `coords_2d`: 1D `torch.float32` tensor of shape `(2,)`.
      - `label`: scalar `torch.long` tensor with spiral branch index in `{0, 1}`.
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
