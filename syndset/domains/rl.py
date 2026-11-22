"""Synthetic reinforcement learning environments and offline trajectory datasets in pure PyTorch."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


class CartPoleMDP:
  """Pure PyTorch vectorized CartPole environment implementing analytical Newtonian mechanics.

  State vector: `[x, x_dot, theta, theta_dot]`
    - `x`: Cart horizontal position (m)
    - `x_dot`: Cart horizontal velocity (m/s)
    - `theta`: Pole angle from vertical (radians)
    - `theta_dot`: Pole angular velocity (rad/s)
  """

  def __init__(
    self,
    gravity: float = 9.8,
    mass_cart: float = 1.0,
    mass_pole: float = 0.1,
    length: float = 0.5,
    force_mag: float = 10.0,
    tau: float = 0.02,
    seed: int = 42,
  ):
    """Initializes CartPole physical parameters.

    Args:
      gravity: Gravitational acceleration (default: 9.8).
      mass_cart: Mass of the cart (default: 1.0).
      mass_pole: Mass of the pole (default: 0.1).
      length: Half-length of the pole (default: 0.5).
      force_mag: Magnitude of applied control force (default: 10.0).
      tau: Discrete integration time step in seconds (default: 0.02).
      seed: Random seed for initialization (default: 42).
    """
    self.gravity = gravity
    self.mass_cart = mass_cart
    self.mass_pole = mass_pole
    self.total_mass = mass_cart + mass_pole
    self.length = length
    self.polemass_length = mass_pole * length
    self.force_mag = force_mag
    self.tau = tau

    self.theta_threshold_radians = 12.0 * 2.0 * math.pi / 360.0
    self.x_threshold = 2.4

    self._generator = torch.Generator().manual_seed(seed)
    self.state = None

  def reset(self) -> torch.Tensor:
    """Resets the environment state uniformly in `[-0.05, 0.05]`."""
    noise = (torch.rand(4, generator=self._generator) - 0.5) * 0.1
    self.state = noise
    return self.state.clone()

  def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict]:
    """Steps the environment forward by one discrete time step tau.

    Args:
      action: Integer in {0: push left, 1: push right}.

    Returns:
      Tuple of (next_state, reward, done, info).
    """
    x, x_dot, theta, theta_dot = self.state.tolist()
    force = self.force_mag if action == 1 else -self.force_mag

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
    thetaacc = (self.gravity * sintheta - costheta * temp) / (
      self.length * (4.0 / 3.0 - self.mass_pole * costheta**2 / self.total_mass)
    )
    xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

    # Euler integration
    x = x + self.tau * x_dot
    x_dot = x_dot + self.tau * xacc
    theta = theta + self.tau * theta_dot
    theta_dot = theta_dot + self.tau * thetaacc

    self.state = torch.tensor([x, x_dot, theta, theta_dot], dtype=torch.float32)

    done = bool(
      x < -self.x_threshold
      or x > self.x_threshold
      or theta < -self.theta_threshold_radians
      or theta > self.theta_threshold_radians
    )
    reward = 1.0 if not done else 0.0
    return self.state.clone(), reward, done, {}


@register_dataset("rl/cartpole_offline")
@register_dataset("rl/cartpole")
class CartPoleOfflineDataset(SyntheticDataset):
  """Generates pre-recorded offline RL trajectory buffers from CartPole simulations.

  Supports evaluating Offline RL algorithms (CQL, IQL, Decision Transformers)
  under controllable behavioral policies: 'expert', 'suboptimal', or 'random'.
  """

  def __init__(
    self,
    num_samples: int = 1000,
    policy: str = "expert",
    horizon: int = 50,
    seed: int = 42,
  ):
    """Initializes offline RL trajectory dataset.

    Args:
      num_samples: Number of transition tuples `(s, a, r, s', done)` (default: 1000).
      policy: Behavioral policy generating data ('expert', 'suboptimal', 'random').
      horizon: Maximum trajectory length before reset (default: 50).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._policy = policy
    self._horizon = horizon

    env = CartPoleMDP(seed=seed)
    generator = torch.Generator().manual_seed(seed)

    states_list = []
    actions_list = []
    rewards_list = []
    next_states_list = []
    dones_list = []

    s = env.reset()
    ep_step = 0

    for _ in range(num_samples):
      # Select action based on policy
      if policy == "random":
        a = torch.randint(0, 2, (1,), generator=generator).item()
      elif policy == "expert":
        # Heuristic balancing PD controller: push towards correcting theta & theta_dot
        theta = s[2].item()
        theta_dot = s[3].item()
        x = s[0].item()
        x_dot = s[1].item()
        val = theta + 0.1 * theta_dot + 0.01 * x + 0.01 * x_dot
        a = 1 if val > 0 else 0
      else:  # Suboptimal: 70% expert, 30% random
        if torch.rand(1, generator=generator).item() < 0.7:
          theta = s[2].item()
          theta_dot = s[3].item()
          a = 1 if (theta + 0.1 * theta_dot) > 0 else 0
        else:
          a = torch.randint(0, 2, (1,), generator=generator).item()

      s_next, r, done, _ = env.step(a)
      ep_step += 1

      if ep_step >= horizon:
        done = True

      states_list.append(s)
      actions_list.append(a)
      rewards_list.append(r)
      next_states_list.append(s_next)
      dones_list.append(1.0 if done else 0.0)

      if done:
        s = env.reset()
        ep_step = 0
      else:
        s = s_next

    self._states = torch.stack(states_list, dim=0)
    self._actions = torch.tensor(actions_list, dtype=torch.long)
    self._rewards = torch.tensor(rewards_list, dtype=torch.float32)
    self._next_states = torch.stack(next_states_list, dim=0)
    self._dones = torch.tensor(dones_list, dtype=torch.float32)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns an offline transition tuple DataSample."""
    return DataSample(
      state=self._states[idx],
      action=self._actions[idx],
      reward=self._rewards[idx],
      next_state=self._next_states[idx],
      done=self._dones[idx],
    )

  def description(self) -> str:
    """Returns a description of the offline RL dataset."""
    return f"CartPole Offline Transitions (policy='{self._policy}', {self._num_samples} steps)"


@register_dataset("rl/gridworld")
class GridworldDataset(SyntheticDataset):
  """Procedurally generates discrete N x N gridworld navigation trajectories.

  Evaluates state representation, value iteration, and shortest path planning.
  Each state is represented by (row, col) coordinates normalized to `[0, 1]`.
  """

  def __init__(
    self,
    num_samples: int = 500,
    grid_size: int = 5,
    seed: int = 42,
  ):
    """Initializes gridworld trajectory dataset.

    Args:
      num_samples: Number of transition steps (default: 500).
      grid_size: Side length of the grid (default: 5).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._grid_size = grid_size

    generator = torch.Generator().manual_seed(seed)
    states = []
    actions = []
    rewards = []
    next_states = []

    goal = (grid_size - 1, grid_size - 1)
    curr_r, curr_c = 0, 0

    # Actions: 0: up, 1: down, 2: left, 3: right
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]

    for _ in range(num_samples):
      s_tensor = torch.tensor([curr_r / grid_size, curr_c / grid_size], dtype=torch.float32)
      a = torch.randint(0, 4, (1,), generator=generator).item()

      nr = max(0, min(grid_size - 1, curr_r + dr[a]))
      nc = max(0, min(grid_size - 1, curr_c + dc[a]))

      r = 1.0 if (nr, nc) == goal else 0.0
      ns_tensor = torch.tensor([nr / grid_size, nc / grid_size], dtype=torch.float32)

      states.append(s_tensor)
      actions.append(a)
      rewards.append(r)
      next_states.append(ns_tensor)

      if (nr, nc) == goal:
        curr_r, curr_c = 0, 0
      else:
        curr_r, curr_c = nr, nc

    self._states = torch.stack(states, dim=0)
    self._actions = torch.tensor(actions, dtype=torch.long)
    self._rewards = torch.tensor(rewards, dtype=torch.float32)
    self._next_states = torch.stack(next_states, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a Gridworld step DataSample."""
    return DataSample(
      state=self._states[idx],
      action=self._actions[idx],
      reward=self._rewards[idx],
      next_state=self._next_states[idx],
    )

  def description(self) -> str:
    """Returns a description of the Gridworld dataset."""
    return f"Gridworld Navigation ({self._grid_size}x{self._grid_size} grid)"
