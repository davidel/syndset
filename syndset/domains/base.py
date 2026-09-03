"""Base classes for synthetic dataset generators."""

from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
  """Base class for all synthetic benchmark datasets in syndset.

  Provides standardized seed management, tensor type casting,
  and descriptive metadata for reporting.
  """

  def __init__(self, num_samples=1000, seed=42):
    """Initializes common dataset properties.

    Args:
      num_samples: Total number of synthetic examples in the dataset (default: 1000).
      seed: Random seed for reproducible generation (default: 42).
    """
    self.num_samples = num_samples
    self.seed = seed

  def __len__(self):
    """Returns the total number of samples."""
    return self.num_samples

  def __getitem__(self, idx):
    """Retrieves a single sample. Must be implemented by subclasses.

    Args:
      idx: Integer index of the sample.

    Raises:
      NotImplementedError: If not implemented in the derived subclass.
    """
    raise NotImplementedError("Subclasses must implement __getitem__.")

  def description(self):
    """Returns a short human-readable description of the benchmark task."""
    return self.__class__.__name__
