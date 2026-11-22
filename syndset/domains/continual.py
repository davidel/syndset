"""Synthetic datasets for continual learning, distribution shift, and shortcut learning."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


@register_dataset("continual/rotating_manifold")
@register_dataset("continual/rotating_stream")
class RotatingManifoldStreamDataset(SyntheticDataset):
  """Evaluates catastrophic forgetting across sequential tasks with rotating feature manifolds.

  Generates a sequence of T tasks where each task rotates a binary classification
  feature distribution by an angle theta_t = t * pi / T.
  """

  def __init__(
    self,
    num_samples_per_task: int = 200,
    num_tasks: int = 5,
    dim: int = 2,
    seed: int = 42,
  ):
    """Initializes rotating manifold stream dataset.

    Args:
      num_samples_per_task: Number of examples per individual task (default: 200).
      num_tasks: Number of sequential tasks (default: 5).
      dim: Dimensionality of feature space (default: 2).
      seed: Random seed for reproducible generation (default: 42).
    """
    total_samples = num_samples_per_task * num_tasks
    super().__init__(num_samples=total_samples, seed=seed)
    self._num_samples_per_task = num_samples_per_task
    self._num_tasks = num_tasks
    self._dim = dim

    generator = torch.Generator().manual_seed(seed)
    features_list = []
    labels_list = []
    tasks_list = []

    for task_id in range(num_tasks):
      angle = task_id * (math.pi / num_tasks)
      cos_a, sin_a = math.cos(angle), math.sin(angle)
      rot_mat = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=torch.float32)

      # Binary classification: mode 0 at (1, 0), mode 1 at (-1, 0)
      for _ in range(num_samples_per_task):
        label = torch.randint(0, 2, (1,), generator=generator).item()
        base_center = torch.tensor([1.0 if label == 1 else -1.0, 0.0], dtype=torch.float32)
        base_point = base_center + torch.randn(2, generator=generator) * 0.3

        rotated_point = rot_mat @ base_point
        features_list.append(rotated_point)
        labels_list.append(label)
        tasks_list.append(task_id)

    self._features = torch.stack(features_list, dim=0)
    self._labels = torch.tensor(labels_list, dtype=torch.long)
    self._tasks = torch.tensor(tasks_list, dtype=torch.long)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing rotated feature vector, label, and task ID."""
    return DataSample(
      features=self._features[idx],
      label=self._labels[idx],
      task_id=self._tasks[idx],
    )

  def description(self) -> str:
    """Returns a description of the rotating stream dataset."""
    desc = f"{self._num_tasks} tasks, {self._num_samples_per_task} per task"
    return f"Rotating Manifold Stream ({desc})"


@register_dataset("continual/spurious_correlation")
@register_dataset("continual/shortcut")
class SpuriousCorrelationDataset(SyntheticDataset):
  """Evaluates shortcut learning and out-of-distribution generalization.

  Features are partitioned into x = [x_core, x_spurious].
  In the 'train' split, x_spurious aligns with label y with high probability
  (e.g. 95%), while in the 'test' split the alignment is inverted (e.g. 5%).
  """

  def __init__(
    self,
    num_samples: int = 1000,
    split: str = "train",
    correlation: float = 0.95,
    seed: int = 42,
  ):
    """Initializes spurious correlation dataset.

    Args:
      num_samples: Number of samples to generate (default: 1000).
      split: Split partition ('train' or 'test').
      correlation: Spurious alignment probability in train split (default: 0.95).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._split = split
    align_prob = correlation if split == "train" else (1.0 - correlation)

    generator = torch.Generator().manual_seed(seed)
    features_list = []
    labels_list = []

    for _ in range(num_samples):
      # True binary label
      y = torch.randint(0, 2, (1,), generator=generator).item()

      # Core causal feature: x_core ~ N(mu_y, 0.5^2)
      x_core = (1.0 if y == 1 else -1.0) + torch.randn(1, generator=generator).item() * 0.4

      # Spurious shortcut feature: matches y with probability align_prob
      if torch.rand(1, generator=generator).item() < align_prob:
        spurious_val = 1.0 if y == 1 else -1.0
      else:
        spurious_val = -1.0 if y == 1 else 1.0
      x_spurious = spurious_val + torch.randn(1, generator=generator).item() * 0.1

      feat = torch.tensor([x_core, x_spurious], dtype=torch.float32)
      features_list.append(feat)
      labels_list.append(y)

    self._features = torch.stack(features_list, dim=0)
    self._labels = torch.tensor(labels_list, dtype=torch.long)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing [x_core, x_spurious] vector and true label."""
    return DataSample(
      features=self._features[idx],
      label=self._labels[idx],
    )

  def description(self) -> str:
    """Returns a description of the spurious correlation dataset."""
    return f"Spurious Correlation (split='{self._split}')"
