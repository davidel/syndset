"""Base classes, data containers, and registry for synthetic dataset generators."""

from collections.abc import Callable
from typing import Any

from torch.utils.data import DataLoader, Dataset


class DataSample(tuple):
  """Container supporting tuple-unpacking, dictionary access, and attribute access.

  Allows legacy tuple unpacking `inputs, targets = sample` while simultaneously
  supporting dictionary access `sample['image']` and attribute access `sample.image`.
  """

  def __new__(cls, *args, **kwargs):
    """Constructs a DataSample instance.

    Args:
      *args: Positional tensor elements.
      **kwargs: Named field mappings. If args is empty and kwargs is provided,
        positional elements are initialized from kwargs values.
    """
    if kwargs and not args:
      inst = super().__new__(cls, tuple(kwargs.values()))
      inst._mapping = dict(kwargs)
    else:
      inst = super().__new__(cls, args)
      inst._mapping = dict(kwargs.get("fields", {}))
    return inst

  def __getitem__(self, item):
    """Retrieves an item by integer index or named string key."""
    if isinstance(item, str):
      if item in self._mapping:
        return self._mapping[item]
      raise KeyError(f"Key '{item}' not found in DataSample fields.")
    return super().__getitem__(item)

  def __getattr__(self, name: str) -> Any:
    """Provides attribute-style access to named fields."""
    if "_mapping" in self.__dict__ and name in self._mapping:
      return self._mapping[name]
    raise AttributeError(f"'DataSample' has no attribute '{name}'")

  def keys(self):
    """Returns field keys."""
    return self._mapping.keys()

  def values(self):
    """Returns field values."""
    return self._mapping.values()

  def items(self):
    """Returns field (key, value) pairs."""
    return self._mapping.items()

  def to_dict(self) -> dict[str, Any]:
    """Converts the sample into a standard Python dictionary."""
    return dict(self._mapping)

  def __repr__(self) -> str:
    """Returns a readable representation of the sample."""
    fields_str = ", ".join(f"{k}={v.__class__.__name__}" for k, v in self._mapping.items())
    return f"DataSample({fields_str})"


_DATASET_REGISTRY: dict[str, Callable[..., Dataset]] = {}


def register_dataset(name: str):
  """Decorator to register a synthetic dataset generator class in the global registry."""

  def decorator(cls):
    _DATASET_REGISTRY[name.lower()] = cls
    return cls

  return decorator


def list_datasets() -> list[str]:
  """Returns a sorted list of all registered dataset names."""
  return sorted(list(_DATASET_REGISTRY.keys()))


def load_dataset(name: str, **kwargs) -> Dataset:
  """Factory function to load any registered synthetic dataset by name.

  Args:
    name: Registered dataset name (case-insensitive, e.g. 'multimodal/shapes').
    **kwargs: Keyword arguments forwarded to the dataset constructor.

  Returns:
    An instantiated SyntheticDataset subclass.

  Raises:
    KeyError: If the requested dataset name is not registered.
  """
  canonical = name.lower()
  if canonical not in _DATASET_REGISTRY:
    available = ", ".join(list_datasets())
    raise KeyError(f"Dataset '{name}' is not registered. Available datasets: {available}")
  return _DATASET_REGISTRY[canonical](**kwargs)


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
    self._num_samples = num_samples
    self._seed = seed

  @property
  def num_samples(self):
    """Returns the total number of samples."""
    return self._num_samples

  @property
  def seed(self):
    """Returns the random seed used for generation."""
    return self._seed

  def __len__(self):
    """Returns the total number of samples."""
    return self._num_samples

  def __getitem__(self, idx):
    """Retrieves a single sample. Must be implemented by subclasses.

    Args:
      idx: Integer index of the sample.

    Raises:
      NotImplementedError: If not implemented in the derived subclass.
    """
    raise NotImplementedError("Subclasses must implement __getitem__.")

  def to_dataloader(self, batch_size=32, shuffle=False, **kwargs) -> DataLoader:
    """Wraps this dataset into a standard PyTorch DataLoader.

    Args:
      batch_size: How many samples per batch to load (default: 32).
      shuffle: Set to True to have the data reshuffled at every epoch (default: False).
      **kwargs: Additional parameters forwarded to `DataLoader`.

    Returns:
      A PyTorch `DataLoader` instance.
    """
    return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

  def to_hf_dataset(self):
    """Exports this dataset to a Hugging Face Dataset instance if available.

    Requires `datasets` to be installed.

    Returns:
      A Hugging Face `datasets.Dataset` instance.

    Raises:
      ImportError: If the `datasets` package is not installed.
    """
    try:
      from datasets import Dataset as HFDataset
    except ImportError as exc:
      raise ImportError("Hugging Face `datasets` package is required for `.to_hf_dataset()`. "
                        "Install it via `pip install datasets`.") from exc

    sample_0 = self[0]
    if isinstance(sample_0, DataSample):
      keys = list(sample_0.keys())
      records = {k: [] for k in keys}
      for idx in range(len(self)):
        s = self[idx]
        for k in keys:
          records[k].append(s[k])
      return HFDataset.from_dict(records)
    elif isinstance(sample_0, (tuple, list)):
      records = {"input": [], "target": []}
      for idx in range(len(self)):
        inp, tgt = self[idx]
        records["input"].append(inp)
        records["target"].append(tgt)
      return HFDataset.from_dict(records)
    elif isinstance(sample_0, dict):
      records = {k: [] for k in sample_0.keys()}
      for idx in range(len(self)):
        s = self[idx]
        for k, v in s.items():
          records[k].append(v)
      return HFDataset.from_dict(records)
    else:
      records = {"data": [self[i] for i in range(len(self))]}
      return HFDataset.from_dict(records)

  def description(self):
    """Returns a short human-readable description of the benchmark task."""
    return self.__class__.__name__
