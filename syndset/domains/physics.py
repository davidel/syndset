"""Synthetic physics-informed neural network (PINN) and differential equation datasets."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


@register_dataset("physics/heat_equation")
@register_dataset("physics/pinn_heat")
class HeatEquationDataset(SyntheticDataset):
  """Evaluates Physics-Informed Neural Networks (PINNs) on the 1D Heat Equation.

  Computes exact closed-form Fourier analytical solutions to:
    partial_t u = alpha * partial_xx u
  on spatial domain x in [0, 1] and temporal domain t in [0, 1] with zero boundary conditions:
    u(x, t) = sum_{n=1}^K A_n * sin(n * pi * x) * exp(-alpha * (n * pi)^2 * t)
  """

  def __init__(
    self,
    num_samples: int = 1000,
    alpha: float = 0.05,
    num_modes: int = 3,
    seed: int = 42,
  ):
    """Initializes heat equation dataset.

    Args:
      num_samples: Number of spatio-temporal coordinate pairs (x, t) (default: 1000).
      alpha: Thermal diffusivity coefficient (default: 0.05).
      num_modes: Number of Fourier modes in the initial condition (default: 3).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._alpha = alpha
    self._num_modes = num_modes

    generator = torch.Generator().manual_seed(seed)
    coords_x = torch.rand(num_samples, 1, generator=generator)
    coords_t = torch.rand(num_samples, 1, generator=generator)

    # Fourier coefficients A_n for modes n=1..K
    coeffs = [1.0 / n for n in range(1, num_modes + 1)]

    u_val = torch.zeros_like(coords_x)
    for n_idx, a_n in enumerate(coeffs):
      n = n_idx + 1
      spatial = torch.sin(n * math.pi * coords_x)
      temporal = torch.exp(-alpha * ((n * math.pi)**2) * coords_t)
      u_val += a_n * spatial * temporal

    self._coords = torch.cat([coords_x, coords_t], dim=1)  # (N, 2): (x, t)
    self._target_u = u_val

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing (x, t) coordinate and analytical solution u(x, t)."""
    return DataSample(
      coordinates=self._coords[idx],
      solution=self._target_u[idx],
    )

  def description(self) -> str:
    """Returns a description of the heat equation dataset."""
    return f"1D Heat Equation Fourier Solution (alpha={self._alpha}, {self._num_modes} modes)"


@register_dataset("physics/burgers_equation")
class BurgersEquationDataset(SyntheticDataset):
  """Evaluates neural operators on the non-linear 1D Burgers' differential equation.

  Simulates wave propagation under non-linear advection and viscous dissipation:
    partial_t u + u * partial_x u = nu * partial_xx u
  """

  def __init__(
    self,
    num_samples: int = 500,
    spatial_points: int = 64,
    nu: float = 0.01,
    seed: int = 42,
  ):
    """Initializes Burgers' equation trajectory dataset.

    Args:
      num_samples: Number of initial condition trajectory pairs (default: 500).
      spatial_points: Spatial grid resolution along x in [0, 2*pi] (default: 64).
      nu: Viscosity parameter (default: 0.01).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._spatial_points = spatial_points
    self._nu = nu

    generator = torch.Generator().manual_seed(seed)
    x = torch.linspace(0.0, 2.0 * math.pi, spatial_points)

    u0_list = []
    ut_list = []

    for _ in range(num_samples):
      # Random sinusoidal initial condition: u0(x) = A * sin(k*x + phi)
      amp = torch.rand(1, generator=generator).item() * 0.8 + 0.2
      k = torch.randint(1, 3, (1,), generator=generator).item()
      phi = torch.rand(1, generator=generator).item() * 2.0 * math.pi
      u0 = amp * torch.sin(k * x + phi)

      # Damped forward approximation after small dt
      dt = 0.1
      # Simple explicit finite-difference step for synthetic target
      dx = 2.0 * math.pi / spatial_points
      u_x = (torch.roll(u0, -1) - torch.roll(u0, 1)) / (2.0 * dx)
      u_xx = (torch.roll(u0, -1) - 2.0 * u0 + torch.roll(u0, 1)) / (dx**2)
      u_next = u0 + dt * (-u0 * u_x + nu * u_xx)

      u0_list.append(u0.unsqueeze(0))
      ut_list.append(u_next.unsqueeze(0))

    self._u0 = torch.stack(u0_list, dim=0)
    self._ut = torch.stack(ut_list, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing initial condition u0 and future state u_t."""
    return DataSample(
      initial_state=self._u0[idx],
      future_state=self._ut[idx],
    )

  def description(self) -> str:
    """Returns a description of the Burgers' dataset."""
    return f"1D Viscous Burgers Equation (nu={self._nu}, {self._spatial_points} spatial pts)"
