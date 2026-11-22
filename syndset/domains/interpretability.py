"""Synthetic interpretability datasets: modular arithmetic and feature superposition for SAEs."""

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


@register_dataset("interpretability/modular_arithmetic")
@register_dataset("interpretability/grokking")
class ModularArithmeticDataset(SyntheticDataset):
  """Evaluates generalization dynamics and grokking on modular arithmetic (a + b) mod p.

  Because modular addition exhibits discrete symmetry under cyclic group Z_p,
  models learn circular Fourier representations in attention and MLP layers.
  """

  def __init__(
    self,
    prime: int = 67,
    operation: str = "add",
    seed: int = 42,
  ):
    """Initializes modular arithmetic dataset.

    Args:
      prime: Prime modulus p (default: 67).
      operation: Binary operation ('add' for (a+b)%p, 'mul' for (a*b)%p).
      seed: Random seed for reproducible generation (default: 42).
    """
    total_pairs = prime * prime
    super().__init__(num_samples=total_pairs, seed=seed)
    self._prime = prime
    self._operation = operation

    generator = torch.Generator().manual_seed(seed)
    pairs = []
    targets = []

    for a in range(prime):
      for b in range(prime):
        if operation == "add":
          ans = (a + b) % prime
        else:
          ans = (a * b) % prime
        pairs.append((a, b))
        targets.append(ans)

    # Permute pairs reproducibly
    perm = torch.randperm(total_pairs, generator=generator)
    pairs_tensor = torch.tensor(pairs, dtype=torch.long)[perm]
    targets_tensor = torch.tensor(targets, dtype=torch.long)[perm]

    self._inputs = pairs_tensor
    self._targets = targets_tensor

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing integer pair (a, b) and modular result."""
    return DataSample(
      inputs=self._inputs[idx],
      target=self._targets[idx],
    )

  def description(self) -> str:
    """Returns a description of the modular arithmetic dataset."""
    return f"Modular Arithmetic ({self._operation}, p={self._prime}, {self._num_samples} pairs)"


@register_dataset("interpretability/feature_superposition")
@register_dataset("interpretability/sae")
class FeatureSuperpositionDataset(SyntheticDataset):
  """Simulates feature superposition in neural networks for Sparse Autoencoder (SAE) evaluation.

  Generates M true underlying sparse features compressed into a d << M dimensional
  linear bottleneck via an almost-orthogonal projection matrix (Johnson-Lindenstrauss).
  Tests dictionary learning and sparse feature recovery in SAEs.
  """

  def __init__(
    self,
    num_samples: int = 1000,
    num_features: int = 64,
    bottleneck_dim: int = 16,
    feature_prob: float = 0.05,
    seed: int = 42,
  ):
    """Initializes feature superposition dataset.

    Args:
      num_samples: Number of activation vectors to generate (default: 1000).
      num_features: Total number of ground-truth monosemantic features M (default: 64).
      bottleneck_dim: Compressed neural activation dimension d << M (default: 16).
      feature_prob: Probability that any feature is active in a given sample (default: 0.05).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._num_features = num_features
    self._bottleneck_dim = bottleneck_dim

    generator = torch.Generator().manual_seed(seed)

    # Near-orthogonal projection dictionary W in R^(d x M)
    dict_w = torch.randn(bottleneck_dim, num_features, generator=generator)
    dict_w = dict_w / torch.norm(dict_w, p=2, dim=0, keepdim=True)
    self._dictionary = dict_w

    # Sparse feature activations h ~ Bernoulli(p) * Uniform(0.5, 2.0)
    is_active = (torch.rand(num_samples, num_features, generator=generator) < feature_prob).float()
    magnitudes = torch.rand(num_samples, num_features, generator=generator) * 1.5 + 0.5
    h_features = is_active * magnitudes  # (num_samples, M)

    # Superposition activations: x = W @ h in R^d
    superposed_x = (dict_w @ h_features.T).T  # (num_samples, d)

    self._superposed_x = superposed_x
    self._true_features = h_features

  @property
  def dictionary(self) -> torch.Tensor:
    """Returns the ground-truth feature dictionary W of shape `(d, M)`."""
    return self._dictionary

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing bottleneck activation and true sparse features."""
    return DataSample(
      activation=self._superposed_x[idx],
      true_features=self._true_features[idx],
    )

  def description(self) -> str:
    """Returns a description of the superposition dataset."""
    return f"Feature Superposition ({self._num_features} features -> R^{self._bottleneck_dim})"
