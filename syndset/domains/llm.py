"""Synthetic sequence datasets for evaluating LLM and autoregressive architectures."""

import torch

from syndset.domains.base import SyntheticDataset


class AssociativeRecallDataset(SyntheticDataset):
  """Evaluates key-value associative retrieval across sequence horizons.

  In language modeling and long-context architectures (Transformers, SSMs,
  Mamba, RWKV), associative recall is the gold-standard test for whether
  the model can bind a key to a value and retrieve it when queried later.

  Sequence structure:
    [k1, v1, k2, v2, ..., kn, vn, query_key] -> target is query_value.
  """

  def __init__(self, num_samples=1000, num_pairs=8, vocab_size=64, seed=42):
    """Initializes the associative recall dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      num_pairs: Number of distinct (key, value) pairs per sequence (default: 8).
      vocab_size: Total vocabulary size. Keys and values are partitioned to avoid overlap.
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self.num_pairs = num_pairs
    self.vocab_size = vocab_size

    # Split vocabulary into distinct keys and values to prevent ambiguity
    self.half_vocab = vocab_size // 2
    if self.half_vocab < num_pairs:
      raise ValueError("vocab_size must be at least 2 * num_pairs.")

    generator = torch.Generator().manual_seed(seed)
    self.inputs = []
    self.targets = []

    for _ in range(num_samples):
      # Sample unique keys from [1, half_vocab]
      key_perm = torch.randperm(self.half_vocab - 1, generator=generator) + 1
      keys = key_perm[:num_pairs]

      # Sample values from [half_vocab + 1, vocab_size - 1]
      val_pool = torch.randint(
        self.half_vocab + 1, self.vocab_size, (num_pairs,), generator=generator
      )

      # Build interleaved sequence: [k1, v1, k2, v2, ...]
      interleaved = torch.empty(num_pairs * 2, dtype=torch.long)
      interleaved[0::2] = keys
      interleaved[1::2] = val_pool

      # Pick one key to query at the end
      query_idx = torch.randint(0, num_pairs, (1,), generator=generator).item()
      query_key = keys[query_idx]
      target_val = val_pool[query_idx]

      # Sequence = [k1, v1, ..., kn, vn, query_key]
      seq_input = torch.cat([interleaved, torch.tensor([query_key], dtype=torch.long)])

      self.inputs.append(seq_input)
      self.targets.append(target_val)

    self.inputs = torch.stack(self.inputs, dim=0)
    self.targets = torch.stack(self.targets, dim=0)

  def __getitem__(self, idx):
    """Returns the input sequence and target token value."""
    return self.inputs[idx], self.targets[idx]

  def description(self):
    """Returns a description of the task."""
    desc_str = f"{self.num_pairs} pairs, seq_len={self.num_pairs * 2 + 1}"
    return f"Associative Recall ({desc_str}, vocab={self.vocab_size})"


class InductionDataset(SyntheticDataset):
  """Evaluates in-context induction head behavior (A ... B ... A -> B).

  Induction heads are the fundamental attention mechanism in LLMs for
  in-context pattern replication. Sequences contain random tokens, where
  a specific prefix pattern [A, B] appears early in the sequence, and
  token [A] appears again at the prompt end, requiring the model to predict [B].
  """

  def __init__(self, num_samples=1000, seq_len=32, vocab_size=64, seed=42):
    """Initializes the induction dataset.

    Args:
      num_samples: Total sequences to generate (default: 1000).
      seq_len: Total length of each sequence (default: 32).
      vocab_size: Vocabulary size (default: 64).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self.seq_len = seq_len
    self.vocab_size = vocab_size

    generator = torch.Generator().manual_seed(seed)
    self.inputs = []
    self.targets = []

    for _ in range(num_samples):
      tokens = torch.randint(1, vocab_size, (seq_len,), generator=generator)

      # Pick token A and token B
      token_a = tokens[0].item()
      token_b = tokens[1].item()

      # Place token A at the second-to-last position
      tokens[-1] = token_a

      self.inputs.append(tokens)
      self.targets.append(torch.tensor(token_b, dtype=torch.long))

    self.inputs = torch.stack(self.inputs, dim=0)
    self.targets = torch.stack(self.targets, dim=0)

  def __getitem__(self, idx):
    """Returns the input sequence and the induction target token."""
    return self.inputs[idx], self.targets[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Induction Head Task (seq_len={self.seq_len}, vocab={self.vocab_size})"


class SelectiveCopyDataset(SyntheticDataset):
  """Evaluates selective memory filtering (copying targets, ignoring noise).

  In this task, the model receives a sequence containing signal tokens
  and distractor/noise tokens. The model must ignore distractors and reproduce
  the signal tokens in exact order.
  """

  def __init__(self, num_samples=1000, num_signals=4, total_len=32, vocab_size=32, seed=42):
    """Initializes the selective copy dataset.

    Args:
      num_samples: Number of sequences (default: 1000).
      num_signals: Number of signal tokens to remember (default: 4).
      total_len: Total sequence length (default: 32).
      vocab_size: Vocabulary size for signal tokens (default: 32).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self.num_signals = num_signals
    self.total_len = total_len
    self.vocab_size = vocab_size

    # Token 0 is reserved for noise/blank distractor
    generator = torch.Generator().manual_seed(seed)
    self.inputs = []
    self.targets = []

    for _ in range(num_samples):
      seq = torch.zeros(total_len, dtype=torch.long)
      # Random positions for signal tokens in the first half
      positions = torch.randperm(total_len // 2, generator=generator)[:num_signals]
      positions, _ = torch.sort(positions)

      signals = torch.randint(1, vocab_size, (num_signals,), generator=generator)
      seq[positions] = signals

      self.inputs.append(seq)
      self.targets.append(signals)

    self.inputs = torch.stack(self.inputs, dim=0)
    self.targets = torch.stack(self.targets, dim=0)

  def __getitem__(self, idx):
    """Returns the noisy sequence and the clean signal sequence to reproduce."""
    return self.inputs[idx], self.targets[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Selective Copy ({self.num_signals} signals in len {self.total_len})"


class CumulativeParityDataset(SyntheticDataset):
  """Evaluates cumulative parity (XOR over time) to test deep compositionality.

  At each time step t, the target is the cumulative XOR of all bits up to t.
  This is a famously challenging task for recurrent networks and Transformers
  because it requires tracking discrete state transitions without degradation.
  """

  def __init__(self, num_samples=1000, seq_len=32, seed=42):
    """Initializes the cumulative parity dataset.

    Args:
      num_samples: Number of sequences (default: 1000).
      seq_len: Length of the binary sequence (default: 32).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self.seq_len = seq_len

    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (num_samples, seq_len), generator=generator)
    cumulative_sum = torch.cumsum(bits, dim=1)
    parity = cumulative_sum % 2

    self.inputs = bits.float()
    self.targets = parity.long()

  def __getitem__(self, idx):
    """Returns binary sequence inputs and step-by-step parity targets."""
    return self.inputs[idx], self.targets[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Cumulative Parity (seq_len={self.seq_len})"
