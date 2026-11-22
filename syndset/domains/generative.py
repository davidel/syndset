"""Synthetic datasets for diffusion, flow matching, score matching, and generative manifolds."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


@register_dataset("generative/score_matching")
@register_dataset("generative/score")
class AnalyticalScoreDataset(SyntheticDataset):
  """Evaluates score-based generative models against exact analytical score functions.

  Constructs a Gaussian Mixture Model in R^dim with analytical density p(x)
  and exact ground-truth score vector field:
    nabla_x log p(x) = sum_k gamma_k(x) * (-(x - mu_k) / sigma_k^2)
  """

  def __init__(
    self,
    num_samples: int = 1000,
    dim: int = 2,
    num_components: int = 4,
    seed: int = 42,
  ):
    """Initializes analytical score dataset.

    Args:
      num_samples: Total number of points (default: 1000).
      dim: Dimensionality of the point space (default: 2).
      num_components: Number of Gaussian modes in the mixture (default: 4).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._dim = dim
    self._num_components = num_components

    generator = torch.Generator().manual_seed(seed)

    # Component centers on a circle/hypersphere of radius 2.0
    end_angle = 2.0 * math.pi * (num_components - 1) / num_components
    angles = torch.linspace(0.0, end_angle, num_components)
    centers = torch.zeros((num_components, dim), dtype=torch.float32)
    centers[:, 0] = torch.cos(angles) * 2.0
    if dim > 1:
      centers[:, 1] = torch.sin(angles) * 2.0
    sigma = 0.5
    self._sigma = sigma

    # Sample points from the mixture
    points = []
    scores = []

    for _ in range(num_samples):
      k = torch.randint(0, num_components, (1,), generator=generator).item()
      mu_k = centers[k]
      x = mu_k + torch.randn(dim, generator=generator) * sigma

      # Compute analytical responsibilities gamma_j(x)
      diffs = x.unsqueeze(0) - centers  # (K, dim)
      dist_sq = torch.sum(diffs**2, dim=-1)  # (K,)
      log_probs = -0.5 * dist_sq / (sigma**2)
      probs = torch.softmax(log_probs, dim=0)  # (K,)

      # Analytical score = sum_j probs[j] * (-(x - mu_j) / sigma^2)
      score = torch.sum(-diffs / (sigma**2) * probs.unsqueeze(-1), dim=0)

      points.append(x)
      scores.append(score)

    self._points = torch.stack(points, dim=0)
    self._scores = torch.stack(scores, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing point coordinate and exact analytical score."""
    return DataSample(
      point=self._points[idx],
      score=self._scores[idx],
    )

  def description(self) -> str:
    """Returns a description of the analytical score dataset."""
    return f"Analytical Score Matching (GMM {self._num_components} modes in R^{self._dim})"


@register_dataset("generative/flow_matching")
@register_dataset("generative/flow")
class FlowMatchingDataset(SyntheticDataset):
  """Generates optimal transport conditional vector fields for Flow Matching.

  Given source x0 ~ N(0, I) and target x1 ~ 8-Gaussians ring, samples intermediate
  state x_t = (1 - t) * x0 + t * x1 at time t ~ Uniform(0, 1), with ground-truth
  conditional velocity field:
    u_t(x_t | x1) = x1 - x0
  """

  def __init__(
    self,
    num_samples: int = 1000,
    dim: int = 2,
    seed: int = 42,
  ):
    """Initializes flow matching dataset.

    Args:
      num_samples: Total number of flow matching tuples (default: 1000).
      dim: Space dimension (default: 2).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._dim = dim

    generator = torch.Generator().manual_seed(seed)

    x0_list = []
    x1_list = []
    xt_list = []
    t_list = []
    v_list = []

    # Target: 8-Gaussians on radius 3.0
    angles = torch.linspace(0.0, 2.0 * math.pi * 7 / 8, 8)
    centers = torch.zeros((8, dim), dtype=torch.float32)
    centers[:, 0] = torch.cos(angles) * 3.0
    if dim > 1:
      centers[:, 1] = torch.sin(angles) * 3.0

    for _ in range(num_samples):
      x0 = torch.randn(dim, generator=generator)
      k = torch.randint(0, 8, (1,), generator=generator).item()
      x1 = centers[k] + torch.randn(dim, generator=generator) * 0.2

      t = torch.rand(1, generator=generator)  # scalar in [0, 1]
      xt = (1.0 - t) * x0 + t * x1
      velocity = x1 - x0

      x0_list.append(x0)
      x1_list.append(x1)
      xt_list.append(xt)
      t_list.append(t)
      v_list.append(velocity)

    self._xt = torch.stack(xt_list, dim=0)
    self._t = torch.stack(t_list, dim=0)
    self._velocity = torch.stack(v_list, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing interpolated point, time step, and target velocity."""
    return DataSample(
      x_t=self._xt[idx],
      time=self._t[idx],
      target_velocity=self._velocity[idx],
    )

  def description(self) -> str:
    """Returns a description of the flow matching dataset."""
    return f"Flow Matching OT Vector Field (dim={self._dim})"


@register_dataset("generative/manifolds_2d")
@register_dataset("generative/swiss_roll")
class ToyManifold2DDataset(SyntheticDataset):
  """Generates canonical 2D/3D non-linear manifolds (Swiss roll, Pinwheel, Checkerboard).

  Useful for testing density estimation, Normalizing Flows, and VAEs.
  """

  def __init__(
    self,
    num_samples: int = 1000,
    manifold_type: str = "swiss_roll",
    noise_std: float = 0.05,
    seed: int = 42,
  ):
    """Initializes 2D toy manifold dataset.

    Args:
      num_samples: Number of coordinate samples (default: 1000).
      manifold_type: Type of manifold ('swiss_roll', 'pinwheel', 'checkerboard').
      noise_std: Additive Gaussian noise (default: 0.05).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._manifold_type = manifold_type
    generator = torch.Generator().manual_seed(seed)

    if manifold_type == "swiss_roll":
      # Swiss roll in 2D: r = theta, x = theta * cos(theta), y = theta * sin(theta)
      theta = 1.5 * math.pi * (1.0 + 2.0 * torch.rand(num_samples, generator=generator))
      x = theta * torch.cos(theta) / 10.0
      y = theta * torch.sin(theta) / 10.0
      points = torch.stack([x, y], dim=1)
    elif manifold_type == "pinwheel":
      # 5-arm pinwheel
      radial = torch.rand(num_samples, generator=generator) * 2.0 + 0.2
      arm = torch.randint(0, 5, (num_samples,), generator=generator).float()
      theta = arm * (2.0 * math.pi / 5.0) + radial * 0.5
      x = radial * torch.cos(theta)
      y = radial * torch.sin(theta)
      points = torch.stack([x, y], dim=1)
    else:  # Checkerboard
      u1 = torch.rand(num_samples, generator=generator) * 4.0 - 2.0
      u2 = torch.rand(num_samples, generator=generator) * 4.0 - 2.0
      points = torch.stack([u1, u2], dim=1)

    noise = torch.randn_like(points) * noise_std
    self._points = points + noise

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing 2D manifold coordinate point."""
    return DataSample(point=self._points[idx])

  def description(self) -> str:
    """Returns a description of the manifold dataset."""
    return f"Toy Manifold 2D (type='{self._manifold_type}')"
