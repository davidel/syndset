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
    self._num_pairs = num_pairs
    self._vocab_size = vocab_size

    # Split vocabulary into distinct keys and values to prevent ambiguity
    half_vocab = vocab_size // 2
    if half_vocab < num_pairs:
      raise ValueError("vocab_size must be at least 2 * num_pairs.")

    generator = torch.Generator().manual_seed(seed)
    inputs_list = []
    targets_list = []

    for _ in range(num_samples):
      # Sample unique keys from [1, half_vocab]
      key_perm = torch.randperm(half_vocab - 1, generator=generator) + 1
      keys = key_perm[:num_pairs]

      # Sample values from [half_vocab + 1, vocab_size - 1]
      val_pool = torch.randint(half_vocab + 1, self._vocab_size, (num_pairs,), generator=generator)

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

      inputs_list.append(seq_input)
      targets_list.append(target_val)

    self._inputs = torch.stack(inputs_list, dim=0)
    self._targets = torch.stack(targets_list, dim=0)

  @property
  def num_pairs(self):
    """Returns the number of key-value pairs per sequence."""
    return self._num_pairs

  @property
  def vocab_size(self):
    """Returns the total vocabulary size."""
    return self._vocab_size

  @property
  def inputs(self):
    """Returns the full tensor of generated input sequences."""
    return self._inputs

  @property
  def targets(self):
    """Returns the full tensor of target token values."""
    return self._targets

  def __getitem__(self, idx):
    """Returns the input sequence and target token value."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    desc_str = f"{self._num_pairs} pairs, seq_len={self._num_pairs * 2 + 1}"
    return f"Associative Recall ({desc_str}, vocab={self._vocab_size})"


class MultiQueryAssociativeRecallDataset(SyntheticDataset):
  """Evaluates Multi-Query Associative Recall (MQAR) across sequence horizons.

  While single-query associative recall can often be approximated by
  sub-quadratic state-space or linear attention models, MQAR (Arora et al., 2023)
  queries multiple distinct keys at the sequence end. This benchmark exposes
  the memory capacity trade-off between standard softmax attention and fixed-state
  recurrent/SSM architectures.

  Sequence structure:
    [k1, v1, ..., kn, vn, q1, q2, ..., qm] -> targets are [v_{q1}, ..., v_{qm}].
  """

  def __init__(self, num_samples=1000, num_pairs=8, num_queries=3, vocab_size=64, seed=42):
    """Initializes the multi-query associative recall dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      num_pairs: Number of distinct (key, value) pairs (default: 8).
      num_queries: Number of distinct queries at the prompt end (default: 3).
      vocab_size: Total vocabulary size (default: 64).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._num_pairs = num_pairs
    self._num_queries = num_queries
    self._vocab_size = vocab_size

    if num_queries > num_pairs:
      raise ValueError("num_queries cannot exceed num_pairs.")

    half_vocab = vocab_size // 2
    if half_vocab < num_pairs:
      raise ValueError("vocab_size must be at least 2 * num_pairs.")

    generator = torch.Generator().manual_seed(seed)
    inputs_list = []
    targets_list = []

    for _ in range(num_samples):
      key_perm = torch.randperm(half_vocab - 1, generator=generator) + 1
      keys = key_perm[:num_pairs]

      val_pool = torch.randint(half_vocab + 1, self._vocab_size, (num_pairs,), generator=generator)

      interleaved = torch.empty(num_pairs * 2, dtype=torch.long)
      interleaved[0::2] = keys
      interleaved[1::2] = val_pool

      # Sample distinct query indices
      q_indices = torch.randperm(num_pairs, generator=generator)[:num_queries]
      query_keys = keys[q_indices]
      target_vals = val_pool[q_indices]

      seq_input = torch.cat([interleaved, query_keys])

      inputs_list.append(seq_input)
      targets_list.append(target_vals)

    self._inputs = torch.stack(inputs_list, dim=0)
    self._targets = torch.stack(targets_list, dim=0)

  @property
  def num_pairs(self):
    """Returns the number of key-value pairs."""
    return self._num_pairs

  @property
  def num_queries(self):
    """Returns the number of queries tested at the sequence end."""
    return self._num_queries

  @property
  def vocab_size(self):
    """Returns the total vocabulary size."""
    return self._vocab_size

  @property
  def inputs(self):
    """Returns the input sequences tensor."""
    return self._inputs

  @property
  def targets(self):
    """Returns the multi-query target token values tensor."""
    return self._targets

  def __getitem__(self, idx):
    """Returns input sequence and query target tokens."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    desc = f"{self._num_pairs} pairs, {self._num_queries} queries"
    return f"Multi-Query Associative Recall ({desc}, vocab={self._vocab_size})"


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
    self._seq_len = seq_len
    self._vocab_size = vocab_size

    generator = torch.Generator().manual_seed(seed)
    inputs_list = []
    targets_list = []

    for _ in range(num_samples):
      tokens = torch.randint(1, vocab_size, (seq_len,), generator=generator)

      # Pick token A and token B
      token_a = tokens[0].item()
      token_b = tokens[1].item()

      # Place token A at the second-to-last position
      tokens[-1] = token_a

      inputs_list.append(tokens)
      targets_list.append(torch.tensor(token_b, dtype=torch.long))

    self._inputs = torch.stack(inputs_list, dim=0)
    self._targets = torch.stack(targets_list, dim=0)

  @property
  def seq_len(self):
    """Returns the sequence length of each sample."""
    return self._seq_len

  @property
  def vocab_size(self):
    """Returns the vocabulary size."""
    return self._vocab_size

  @property
  def inputs(self):
    """Returns the tensor of generated token sequences."""
    return self._inputs

  @property
  def targets(self):
    """Returns the tensor of target continuation tokens."""
    return self._targets

  def __getitem__(self, idx):
    """Returns the input sequence and the induction target token."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Induction Head Task (seq_len={self._seq_len}, vocab={self._vocab_size})"


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
    self._num_signals = num_signals
    self._total_len = total_len
    self._vocab_size = vocab_size

    # Token 0 is reserved for noise/blank distractor
    generator = torch.Generator().manual_seed(seed)
    inputs_list = []
    targets_list = []

    for _ in range(num_samples):
      seq = torch.zeros(total_len, dtype=torch.long)
      # Random positions for signal tokens in the first half
      positions = torch.randperm(total_len // 2, generator=generator)[:num_signals]
      positions, _ = torch.sort(positions)

      signals = torch.randint(1, vocab_size, (num_signals,), generator=generator)
      seq[positions] = signals

      inputs_list.append(seq)
      targets_list.append(signals)

    self._inputs = torch.stack(inputs_list, dim=0)
    self._targets = torch.stack(targets_list, dim=0)

  @property
  def num_signals(self):
    """Returns the number of signal tokens per sequence."""
    return self._num_signals

  @property
  def total_len(self):
    """Returns the total sequence length."""
    return self._total_len

  @property
  def vocab_size(self):
    """Returns the vocabulary size."""
    return self._vocab_size

  @property
  def inputs(self):
    """Returns the noisy input token sequences."""
    return self._inputs

  @property
  def targets(self):
    """Returns the target signal sequences."""
    return self._targets

  def __getitem__(self, idx):
    """Returns the noisy sequence and the clean signal sequence to reproduce."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Selective Copy ({self._num_signals} signals in len {self._total_len})"


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
    self._seq_len = seq_len

    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (num_samples, seq_len), generator=generator)
    cumulative_sum = torch.cumsum(bits, dim=1)
    parity = cumulative_sum % 2

    self._inputs = bits.float()
    self._targets = parity.long()

  @property
  def seq_len(self):
    """Returns the binary sequence length."""
    return self._seq_len

  @property
  def inputs(self):
    """Returns the binary input sequences."""
    return self._inputs

  @property
  def targets(self):
    """Returns the cumulative parity targets."""
    return self._targets

  def __getitem__(self, idx):
    """Returns binary sequence inputs and step-by-step parity targets."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Cumulative Parity (seq_len={self._seq_len})"


class DyckLanguageDataset(SyntheticDataset):
  """Evaluates nested stack memory and context-free grammar parsing (Dyck-k).

  Dyck languages consist of well-nested bracket strings (e.g. `(()())`).
  In mechanistic interpretability and formal language theory, Dyck languages
  test whether an architecture can simulate a pushdown automaton with a
  hierarchical stack memory.

  At each position t, the task is to predict the required closing token
  matching the bracket at the top of the stack (or 0 if stack is empty).
  """

  def __init__(self, num_samples=1000, seq_len=32, num_types=2, max_depth=6, seed=42):
    """Initializes the Dyck language dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      seq_len: Length of each bracket sequence (default: 32).
      num_types: Number of bracket pairs (default: 2, e.g. round and square).
      max_depth: Maximum stack nesting depth allowed (default: 6).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._seq_len = seq_len
    self._num_types = num_types
    self._max_depth = max_depth

    generator = torch.Generator().manual_seed(seed)
    inputs_list = []
    targets_list = []

    # Bracket tokens: for type t in 1..num_types:
    #   open_t = 2 * t - 1
    #   close_t = 2 * t
    for _ in range(num_samples):
      seq = []
      targets = []
      stack = []

      for _step in range(seq_len):
        can_open = len(stack) < max_depth
        can_close = len(stack) > 0

        # Decide whether to open or close
        if not can_close:
          action_open = True
        elif not can_open:
          action_open = False
        else:
          action_open = bool(torch.rand(1, generator=generator).item() < 0.5)

        if action_open:
          # Pick bracket type uniformly in 1..num_types
          b_type = torch.randint(1, num_types + 1, (1,), generator=generator).item()
          open_token = 2 * b_type - 1
          close_token = 2 * b_type
          seq.append(open_token)
          stack.append(close_token)
        else:
          close_token = stack.pop()
          seq.append(close_token)

        # Expected next closing token matching current top of stack
        top_expected = stack[-1] if len(stack) > 0 else 0
        targets.append(top_expected)

      inputs_list.append(torch.tensor(seq, dtype=torch.long))
      targets_list.append(torch.tensor(targets, dtype=torch.long))

    self._inputs = torch.stack(inputs_list, dim=0)
    self._targets = torch.stack(targets_list, dim=0)

  @property
  def seq_len(self):
    """Returns the sequence length."""
    return self._seq_len

  @property
  def num_types(self):
    """Returns the number of distinct bracket pairs."""
    return self._num_types

  @property
  def max_depth(self):
    """Returns the maximum nesting depth."""
    return self._max_depth

  @property
  def inputs(self):
    """Returns the tensor of generated bracket token sequences."""
    return self._inputs

  @property
  def targets(self):
    """Returns the tensor of step-by-step stack closure targets."""
    return self._targets

  def __getitem__(self, idx):
    """Returns bracket input sequence and stack top targets."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    desc = f"Dyck-{self._num_types} (len={self._seq_len}, max_depth={self._max_depth})"
    return f"Dyck Language Stack Memory ({desc})"
