"""Synthetic sequence datasets for evaluating LLM and autoregressive architectures."""

import torch

from syndset.domains.base import SyntheticDataset


class AssociativeRecallDataset(SyntheticDataset):
  """Evaluates key-value associative retrieval across sequence horizons.

  In language modeling and long-context architectures (Transformers, SSMs,
  Mamba, RWKV), associative recall is the gold-standard test for whether
  the model can bind a key to a value and retrieve it when queried later.

  What this dataset generates:
    1. Vocabulary Partitioning: The discrete vocabulary {0, ..., vocab_size - 1}
       is split into two disjoint pools:
         - Key pool: tokens in [1, vocab_size // 2]
         - Value pool: tokens in [vocab_size // 2 + 1, vocab_size - 1]
         - Token 0 is reserved/unused. Disjoint sets prevent ambiguous key/value roles.
    2. Sequence Assembly:
       - Generates `num_pairs` distinct keys sampled without replacement from the key pool.
       - Generates `num_pairs` values sampled with replacement from the value pool.
       - Interleaves keys and values: [k_0, v_0, k_1, v_1, ..., k_{n-1}, v_{n-1}].
    3. Query Insertion:
       - Uniformly picks one of the preceding keys: query_key = k_i (for some random i).
       - Appends query_key at prompt end (input sequence length: 2 * num_pairs + 1).
    4. Target Output:
       - The ground-truth target is the corresponding scalar value v_i paired with k_i.

  Sequence Layout:
    inputs[idx]:  [ k_0, v_0, ..., k_{n-1}, v_{n-1}, query_key ] (shape: (2*num_pairs + 1,))
    targets[idx]: query_value                                    (scalar)

  Concrete Example:
    Suppose num_pairs=3, vocab_size=16 (keys: 1..8, values: 9..15):
      Keys sampled:   [3, 7, 2]
      Values sampled: [11, 14, 10]
      Pairs: (3 -> 11), (7 -> 14), (2 -> 10)
      Query chosen: key 7 (paired with value 14)
      Input sequence:  [ 3, 11,  7, 14,  2, 10,  7 ]  (shape: (7,))
      Target value:    14                                (scalar)

  Attributes:
    inputs: Tensor of shape `(num_samples, 2 * num_pairs + 1)` with dtype `torch.long`.
    targets: Tensor of shape `(num_samples,)` with dtype `torch.long`.

  Item Format:
    Indexing `dataset[idx]` returns `(input_seq, target_val)` where:
      - `input_seq`: 1D `torch.long` tensor of shape `(2 * num_pairs + 1,)`.
      - `target_val`: scalar `torch.long` tensor with the true query value.
  """

  def __init__(self, num_samples=1000, num_pairs=8, vocab_size=64, seed=42):
    """Initializes the associative recall dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      num_pairs: Number of distinct (key, value) pairs per sequence (default: 8).
        Total sequence length is `2 * num_pairs + 1`.
      vocab_size: Total vocabulary size partitioned between keys and values (default: 64).
        Must satisfy `vocab_size >= 2 * num_pairs`. Keys are sampled from `[1, vocab_size // 2]`,
        and values are sampled from `[vocab_size // 2 + 1, vocab_size - 1]`.
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

  What this dataset generates:
    1. Vocabulary Partitioning:
       - Key pool: tokens in [1, vocab_size // 2]
       - Value pool: tokens in [vocab_size // 2 + 1, vocab_size - 1]
    2. Sequence Assembly:
       - Generates `num_pairs` distinct keys and `num_pairs` values.
       - Interleaves keys and values into an initial prefix:
         [k_0, v_0, k_1, v_1, ..., k_{n-1}, v_{n-1}] (length: 2 * num_pairs).
    3. Multi-Query Insertion:
       - Samples `num_queries` distinct query keys from {k_0, ..., k_{n-1}} without replacement:
         [q_0, q_1, ..., q_{m-1}].
       - Appends all query keys at the end of the sequence, yielding an input
         sequence of length 2 * num_pairs + num_queries.
    4. Target Output:
       - The target is a 1D tensor of `num_queries` values: [v_{q_0}, v_{q_1}, ..., v_{q_{m-1}}].

  Sequence Layout:
    inputs[idx]:  [ k_0, v_0, ..., k_{n-1}, v_{n-1}, q_0, ..., q_{m-1} ]
                  (shape: (2*num_pairs + num_queries,), int64 in 0..V-1)
    targets[idx]: [ v_{q_0}, v_{q_1}, ..., v_{q_{m-1}} ] (shape: (num_queries,), int64 in 0..V-1)

  Concrete Example:
    Suppose num_pairs=4, num_queries=2, vocab_size=32 (keys: 1..16, values: 17..31):
      Pairs: (5 -> 20), (2 -> 29), (11 -> 18), (8 -> 24)
      Queries selected: [11, 5]
      Input sequence:  [ 5, 20,  2, 29, 11, 18,  8, 24, 11,  5 ]  (shape: (10,))
      Target values:   [ 18, 20 ]                                    (shape: (2,))

  Attributes:
    inputs: Tensor of shape `(num_samples, 2 * num_pairs + num_queries)` with dtype `torch.long`.
    targets: Tensor of shape `(num_samples, num_queries)` with dtype `torch.long`.

  Item Format:
    Indexing `dataset[idx]` returns `(input_seq, target_vals)` where:
      - `input_seq`: 1D `torch.long` tensor of shape `(2 * num_pairs + num_queries,)`.
      - `target_vals`: 1D `torch.long` tensor of shape `(num_queries,)`.
  """

  def __init__(self, num_samples=1000, num_pairs=8, num_queries=3, vocab_size=64, seed=42):
    """Initializes the multi-query associative recall dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      num_pairs: Number of distinct (key, value) pairs (default: 8).
      num_queries: Number of distinct queries at the prompt end (default: 3).
        Must satisfy `num_queries <= num_pairs`.
      vocab_size: Total vocabulary size (default: 64). Must satisfy `vocab_size >= 2 * num_pairs`.
      seed: Random seed for reproducible generation (default: 42).
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

  What this dataset generates:
    1. Base Sequence Generation:
       - Generates a sequence of `seq_len` tokens sampled uniformly from [1, vocab_size - 1].
    2. Induction Trigger Setup:
       - Token A = tokens[0] and Token B = tokens[1] establish the trigger transition A -> B.
       - Distractor noise fills intermediate positions tokens[2 : seq_len - 1].
    3. Target Injection:
       - Token A is re-inserted at the final position: tokens[-1] = token_a.
       - The target label is scalar token B.
       - To solve the task, the model must attend from token A at position -1 back to
         its initial occurrence at position 0, look up the subsequent token B at position 1,
         and output B.

  Sequence Layout:
    inputs[idx]:  [ A, B, r_2, r_3, ..., r_{T-2}, A ] (shape: (seq_len,))
    targets[idx]: B                                   (scalar)

  Concrete Example:
    Suppose seq_len=8, vocab_size=32:
      Initial random tokens: [ 14, 27,  3, 19,  8, 22, 11,  5 ]
      Trigger pattern:        A = 14, B = 27
      Final position updated: tokens[-1] = 14
      Input sequence:        [ 14, 27,  3, 19,  8, 22, 11, 14 ]  (shape: (8,))
      Target continuation:   27                                  (scalar)

  Attributes:
    inputs: Tensor of shape `(num_samples, seq_len)` with dtype `torch.long`.
    targets: Tensor of shape `(num_samples,)` with dtype `torch.long`.

  Item Format:
    Indexing `dataset[idx]` returns `(tokens, target_token)` where:
      - `tokens`: 1D `torch.long` tensor of shape `(seq_len,)`.
      - `target_token`: scalar `torch.long` tensor containing the target token B.
  """

  def __init__(self, num_samples=1000, seq_len=32, vocab_size=64, seed=42):
    """Initializes the induction dataset.

    Args:
      num_samples: Total sequences to generate (default: 1000).
      seq_len: Total length of each sequence (default: 32).
      vocab_size: Vocabulary size (default: 64).
      seed: Random seed for reproducible generation (default: 42).
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

  What this dataset generates:
    1. Distractor Background:
       - Allocates a sequence of length `total_len` filled with distractor tokens (token 0).
    2. Signal Token Sampling:
       - Samples `num_signals` distinct sorted positions in the first half of the sequence:
         positions in [0, total_len // 2).
       - Samples `num_signals` non-zero signal tokens from [1, vocab_size - 1].
    3. Input Sequence Assembly:
       - Places the signal tokens at the selected positions, leaving all other positions as 0.
       - The entire second half of the sequence [total_len // 2 :] is pure padding/noise (token 0).
    4. Target Output:
       - The target is a 1D tensor of `num_signals` tokens representing the signal sequence
         in original temporal order.

  Sequence Layout:
    inputs[idx]:  [ 0, ..., s_0, 0, ..., s_1, 0, ..., 0, 0, 0 ] (shape: (total_len,))
    targets[idx]: [ s_0, s_1, ..., s_{k-1} ]                   (shape: (num_signals,))

  Concrete Example:
    Suppose total_len=10, num_signals=3, vocab_size=16:
      Signal positions chosen in first half (0..4): [1, 2, 4]
      Signal tokens generated: [11, 7, 14]
      Input sequence:  [  0, 11,  7,  0, 14,  0,  0,  0,  0,  0 ]  (shape: (10,))
      Target signals:  [ 11,  7, 14 ]                                (shape: (3,))

  Attributes:
    inputs: Tensor of shape `(num_samples, total_len)` with dtype `torch.long`.
    targets: Tensor of shape `(num_samples, num_signals)` with dtype `torch.long`.

  Item Format:
    Indexing `dataset[idx]` returns `(noisy_seq, signal_seq)` where:
      - `noisy_seq`: 1D `torch.long` tensor of shape `(total_len,)` with distractor zeros.
      - `signal_seq`: 1D `torch.long` tensor of shape `(num_signals,)` with ordered signals.
  """

  def __init__(self, num_samples=1000, num_signals=4, total_len=32, vocab_size=32, seed=42):
    """Initializes the selective copy dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      num_signals: Number of signal tokens to remember (default: 4).
      total_len: Total sequence length (default: 32).
      vocab_size: Vocabulary size for signal tokens (default: 32). Token 0 is reserved for noise.
      seed: Random seed for reproducible generation (default: 42).
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

  What this dataset generates:
    1. Binary Bit Generation:
       - Samples `seq_len` binary bits uniformly from {0, 1}.
    2. Cumulative Step-by-Step XOR Parity:
       - Computes running cumulative sum: cumsum[t] = sum_{i=0}^t bits[i].
       - Computes step parity: targets[t] = cumsum[t] % 2 (equivalent to bits[0] ^ ... ^ bits[t]).
    3. Types and Formats:
       - `inputs`: float32 tensor of bits (0.0 or 1.0) of shape (seq_len,).
       - `targets`: long tensor of parity labels (0 or 1) of shape (seq_len,).

  Sequence Layout:
    inputs[idx]:  [ b_0, b_1, ..., b_{T-1} ]             (shape: (seq_len,), float32 in {0.0, 1.0})
    targets[idx]: [ b_0, b_0^b_1, ..., b_0^...^b_{T-1} ] (shape: (seq_len,), int64 in {0, 1})

  Concrete Example:
    Suppose seq_len=8:
      Input bits (float32): [ 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0 ]
      Running sum:          [ 1,   1,   2,   3,   3,   4,   4,   4   ]
      Target parity (long): [ 1,   1,   0,   1,   1,   0,   0,   0   ]

  Attributes:
    inputs: Tensor of shape `(num_samples, seq_len)` with dtype `torch.float32`.
    targets: Tensor of shape `(num_samples, seq_len)` with dtype `torch.long`.

  Item Format:
    Indexing `dataset[idx]` returns `(bits, parity_targets)` where:
      - `bits`: 1D `torch.float32` tensor of shape `(seq_len,)` with values in `{0.0, 1.0}`.
      - `parity_targets`: 1D `torch.long` tensor of shape `(seq_len,)` with cumulative parity
        values in `{0, 1}`.
  """

  def __init__(self, num_samples=1000, seq_len=32, seed=42):
    """Initializes the cumulative parity dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      seq_len: Length of each binary sequence (default: 32).
      seed: Random seed for reproducible generation (default: 42).
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

  What this dataset generates:
    1. Bracket Token Alphabet:
       - For `num_types` bracket pairs (e.g. round `()`, square `[]`), tokens are encoded as:
         * Type 1: open = 1 `(`, close = 2 `)`
         * Type 2: open = 3 `[`, close = 4 `]`
         * Type t: open = 2t - 1, close = 2t
    2. Grammar-Valid Pushdown Simulation:
       - Maintains an internal runtime stack of expected closing tokens.
       - At each time step:
         * If stack is empty: must push an open bracket.
         * If stack depth reaches `max_depth`: must pop and emit the matching closing bracket.
         * Otherwise: with 50% probability pushes an open bracket, and with 50% probability
           pops and emits the matching close bracket.
    3. Step-by-Step Stack Targets:
       - After each token is emitted, `targets[t]` records the closing token matching the
         CURRENT top of the stack (or 0 if the stack is currently empty).

  Sequence Layout:
    inputs[idx]:  [ b_0, b_1, b_2, ..., b_{T-1} ] (shape: (seq_len,), int64 in 1..2*num_types)
    targets[idx]: [ c_0, c_1, c_2, ..., c_{T-1} ] (shape: (seq_len,), int64 in 0..2*num_types)

  Concrete Example:
    Suppose num_types=2 (round `(1, 2)`, square `[3, 4]`), seq_len=6:
      Step 0: Emit open `(` (token 1). Stack: [2]. Target: 2
      Step 1: Emit open `[` (token 3). Stack: [2, 4]. Target: 4
      Step 2: Emit close `]` (token 4). Stack: [2]. Target: 2
      Step 3: Emit open `[` (token 3). Stack: [2, 4]. Target: 4
      Step 4: Emit close `]` (token 4). Stack: [2]. Target: 2
      Step 5: Emit close `)` (token 2). Stack: []. Target: 0
      Input sequence:  [ 1, 3, 4, 3, 4, 2 ]  (shape: (6,))
      Target sequence: [ 2, 4, 2, 4, 2, 0 ]  (shape: (6,))

  Attributes:
    inputs: Tensor of shape `(num_samples, seq_len)` with dtype `torch.long`.
    targets: Tensor of shape `(num_samples, seq_len)` with dtype `torch.long`.

  Item Format:
    Indexing `dataset[idx]` returns `(seq, targets)` where:
      - `seq`: 1D `torch.long` tensor of shape `(seq_len,)` of bracket tokens.
      - `targets`: 1D `torch.long` tensor of shape `(seq_len,)` of expected closing tokens.
  """

  def __init__(self, num_samples=1000, seq_len=32, num_types=2, max_depth=6, seed=42):
    """Initializes the Dyck language dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      seq_len: Length of each bracket sequence (default: 32).
      num_types: Number of distinct bracket pairs (default: 2, e.g. round and square).
      max_depth: Maximum stack nesting depth allowed (default: 6).
      seed: Random seed for reproducible generation (default: 42).
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


class MarkovLanguageDataset(SyntheticDataset):
  """Evaluates statistical next-token probability modeling and causal validity.

  Generates sequences from an order-k Markov process over a discrete vocabulary
  of size V. Transition probability distributions P(x_t | x_{t-k:t-1}) are
  sampled from a Dirichlet prior with concentration parameter alpha.

  The dataset exposes:
    - Theoretical Bayes-optimal Shannon entropy rate H(X) in nats.
    - Full step-by-step target probability distributions for KL divergence.
    - Exhaustive prefix grid for sub-10ms deterministic distribution auditing.
    - Automated causal leakage detection (loss < H(X) indicates lookahead bug).

  What this dataset generates:
    1. Transition Matrix:
       - Generates a transition matrix of shape (vocab_size**order, vocab_size) where each row
         corresponds to a context tuple (x_{t-k}, ..., x_{t-1}) and defines a categorical
         distribution P(x_t | context) sampled from Dirichlet(alpha * 1_V).
    2. Autoregressive Sequence Sampling:
       - Samples initial k prefix tokens uniformly from [0, vocab_size - 1].
       - For t = k, ..., seq_len: samples next token x_t ~ P(. | x_{t-k : t}).
    3. Input and Target Sequence Shift:
       - Input sequence:  tokens[0 : seq_len] of shape (seq_len,).
       - Target sequence: shifted next tokens tokens[1 : seq_len + 1] of shape (seq_len,).
    4. Target Probability Tensors:
       - Stores full ground-truth conditional distributions P(x_{t+1} | x_{t-k+1:t}) in
         `.target_probs` of shape (num_samples, seq_len, vocab_size) for exact KL divergence
         evaluation.

  Sequence Layout:
    inputs[idx]:       [ x_0, x_1, x_2, ..., x_{T-1} ] (shape: (seq_len,), int64 in 0..V-1)
    targets[idx]:      [ x_1, x_2, x_3, ..., x_T ]     (shape: (seq_len,), int64 in 0..V-1)
    target_probs[idx]: [ P(x_1|x_0), ..., P(x_T|x_{<T}) ]
                       (shape: (seq_len, vocab_size), float32)

  Concrete Example:
    Suppose vocab_size=4, order=2, seq_len=4:
      Initial context: [ 1, 3 ]
      Transition row for (1, 3): [ 0.10, 0.70, 0.05, 0.15 ] -> sampled token 1
      Next context: (3, 1)
      Transition row for (3, 1): [ 0.05, 0.10, 0.80, 0.05 ] -> sampled token 2
      Next context: (1, 2)
      Transition row for (1, 2): [ 0.60, 0.20, 0.10, 0.10 ] -> sampled token 0
      Full sequence:   [ 1, 3, 1, 2, 0 ]
      Input sequence:  [ 1, 3, 1, 2 ]  (shape: (4,))
      Target sequence: [ 3, 1, 2, 0 ]  (shape: (4,))

  Attributes:
    inputs: Tensor of shape `(num_samples, seq_len)` with dtype `torch.long`.
    targets: Tensor of shape `(num_samples, seq_len)` with dtype `torch.long` (next-token targets).
    target_probs: Tensor of shape `(num_samples, seq_len, vocab_size)` with dtype `torch.float32`.
    transition_matrix: Tensor of shape `(vocab_size**order, vocab_size)` with dtype `torch.float32`.
    prefix_grid: Tuple of `(prefixes, prefix_distributions)`.
    theoretical_entropy: Theoretical Shannon entropy floor in nats.
    macro_entropy: Unweighted macro average entropy across all transition rows.

  Item Format:
    Indexing `dataset[idx]` returns `(input_seq, target_seq)` where:
      - `input_seq`: 1D `torch.long` tensor of shape `(seq_len,)`.
      - `target_seq`: 1D `torch.long` tensor of shape `(seq_len,)` of shifted next tokens.
  """

  def __init__(
    self,
    num_samples=1000,
    seq_len=32,
    vocab_size=8,
    order=2,
    alpha=1.0,
    seed=42,
  ):
    """Initializes the Markov language dataset.

    Args:
      num_samples: Number of sequences to generate (default: 1000).
      seq_len: Sequence length for inputs and step-by-step targets (default: 32).
      vocab_size: Total discrete vocabulary size V >= 2 (default: 8).
      order: Markov order k >= 1 (default: 2).
      alpha: Dirichlet concentration parameter for transition distribution prior (default: 1.0).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    if vocab_size < 2:
      raise ValueError("vocab_size must be at least 2.")
    if order < 1:
      raise ValueError("order must be at least 1.")
    if seq_len < order:
      raise ValueError("seq_len must be at least order.")
    if alpha <= 0:
      raise ValueError("alpha must be positive.")
    if vocab_size**order > 1_000_000:
      raise ValueError(f"vocab_size**order ({vocab_size**order}) exceeds limit of 1,000,000.")

    self._vocab_size = vocab_size
    self._order = order
    self._seq_len = seq_len
    self._alpha = alpha

    generator = torch.Generator().manual_seed(seed)
    num_contexts = vocab_size**order
    weights = vocab_size**torch.arange(order - 1, -1, -1, dtype=torch.long)

    # Sample transition distributions from Dirichlet(alpha)
    gamma_samples = torch._standard_gamma(torch.full((num_contexts, vocab_size), alpha),
                                          generator=generator)
    self._transition_matrix = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

    # Precompute prefix grid (all V^k possible context combinations)
    prefixes = torch.cartesian_prod(*[torch.arange(vocab_size) for _ in range(order)])
    if order == 1:
      prefixes = prefixes.unsqueeze(-1)
    self._prefixes = prefixes
    self._prefix_distributions = self._transition_matrix

    # Generate sequence tokens
    # Generate seq_len + 1 tokens: inputs are seq[:seq_len], targets are seq[1:seq_len + 1]
    seq = torch.empty((num_samples, seq_len + 1), dtype=torch.long)
    seq[:, :order] = torch.randint(0, vocab_size, (num_samples, order), generator=generator)

    for t in range(order, seq_len + 1):
      ctx = seq[:, t - order:t]
      c_idx = (ctx * weights).sum(dim=-1)
      seq[:, t] = torch.multinomial(self._transition_matrix[c_idx], 1,
                                    generator=generator).squeeze(-1)

    self._inputs = seq[:, :seq_len]
    self._targets = seq[:, 1:seq_len + 1]

    # Precompute target probability distributions across positions
    target_probs = torch.empty((num_samples, seq_len, vocab_size), dtype=torch.float32)
    uniform_prob = 1.0 / vocab_size
    for t in range(seq_len):
      if t < order - 1:
        target_probs[:, t, :] = uniform_prob
      else:
        ctx = self._inputs[:, t - order + 1:t + 1]
        c_idx = (ctx * weights).sum(dim=-1)
        target_probs[:, t, :] = self._transition_matrix[c_idx]

    self._target_probs = target_probs

    # Expected Bayes-optimal Shannon entropy across generated sequence contexts (in nats)
    eps = 1e-12
    conditioned_probs = target_probs[:, order - 1:, :]
    entropy_conditioned = -torch.sum(conditioned_probs * torch.log(conditioned_probs + eps), dim=-1)
    self._theoretical_entropy = float(entropy_conditioned.mean().item())

    # Unweighted macro average entropy across all transition matrix rows
    entropy_per_context = -torch.sum(
      self._transition_matrix * torch.log(self._transition_matrix + eps), dim=-1)
    self._macro_entropy = float(entropy_per_context.mean().item())

  @property
  def vocab_size(self):
    """Returns the vocabulary size."""
    return self._vocab_size

  @property
  def order(self):
    """Returns the Markov order k."""
    return self._order

  @property
  def seq_len(self):
    """Returns the sequence length."""
    return self._seq_len

  @property
  def alpha(self):
    """Returns the Dirichlet concentration parameter."""
    return self._alpha

  @property
  def inputs(self):
    """Returns the generated input token sequences tensor."""
    return self._inputs

  @property
  def targets(self):
    """Returns the next-token target tensor."""
    return self._targets

  @property
  def target_probs(self):
    """Returns the step-by-step target probability distributions tensor."""
    return self._target_probs

  @property
  def transition_matrix(self):
    """Returns the full ground-truth transition probability matrix."""
    return self._transition_matrix

  @property
  def theoretical_entropy(self):
    """Returns the expected Bayes-optimal Shannon entropy in nats."""
    return self._theoretical_entropy

  @property
  def macro_entropy(self):
    """Returns the unweighted macro average Shannon entropy across all contexts."""
    return self._macro_entropy

  @property
  def prefix_grid(self):
    """Returns a tuple of (all_prefixes, true_distributions)."""
    return self._prefixes, self._prefix_distributions

  def __getitem__(self, idx):
    """Returns the input sequence and step-by-step next-token targets."""
    return self._inputs[idx], self._targets[idx]

  def description(self):
    """Returns a description of the task."""
    desc = (f"order={self._order}, vocab={self._vocab_size}, "
            f"len={self._seq_len}, H={self._theoretical_entropy:.3f} nats")
    return f"Markov Language Task ({desc})"

  def evaluate_distribution(self, model):
    """Evaluates how closely a trained model matches the true Markov distributions.

    Runs the complete prefix grid through the model in a single forward pass
    and computes the mean Total Variation (TV) distance and forward KL divergence.

    Args:
      model: A PyTorch nn.Module taking token inputs of shape (batch, order)
        and returning logits of shape (batch, vocab_size) or (batch, order, vocab_size).

    Returns:
      A dictionary containing:
        - 'status': 'passed' (TV < 0.15), 'marginal' (TV < 0.30), or 'failed'
        - 'mean_tv_distance': Average Total Variation distance in [0, 1]
        - 'mean_kl_divergence': Average KL divergence in nats
        - 'theoretical_entropy': Shannon entropy of the true Markov source
    """
    model.eval()
    with torch.no_grad():
      outputs = model(self._prefixes)
      if hasattr(outputs, "logits"):
        logits = outputs.logits
      elif isinstance(outputs, (tuple, list)):
        logits = outputs[0]
      else:
        logits = outputs

      if logits.dim() == 3:
        logits = logits[:, -1, :]

      pred_probs = torch.softmax(logits, dim=-1)
      true_probs = self._prefix_distributions

      # TV distance: 0.5 * sum(|p - q|)
      tv_dist = 0.5 * torch.sum(torch.abs(pred_probs - true_probs), dim=-1).mean().item()

      # Forward KL divergence: sum(p * log(p / q))
      eps = 1e-12
      kl_div = (torch.sum(true_probs * (torch.log(true_probs + eps) - torch.log(pred_probs + eps)),
                          dim=-1).mean().item())

    if tv_dist < 0.15:
      status = "passed"
    elif tv_dist < 0.30:
      status = "marginal"
    else:
      status = "failed"

    return {
      "status": status,
      "mean_tv_distance": tv_dist,
      "mean_kl_divergence": kl_div,
      "theoretical_entropy": self._theoretical_entropy,
    }

  def check_causal_leakage(self, model, inputs=None, targets=None, tolerance=0.05):
    """Checks whether the model's loss falls below theoretical entropy (causal leakage).

    In an autoregressive setting, no causal model can achieve expected cross-entropy
    lower than the Shannon entropy of the generating Markov source. If empirical
    loss is strictly less than H(X) - tolerance, future tokens are leaking.

    Args:
      model: PyTorch nn.Module to evaluate.
      inputs: Optional input tensor. Defaults to self._inputs.
      targets: Optional target tensor. Defaults to self._targets.
      tolerance: Allowable margin below theoretical entropy before flagging leakage.

    Returns:
      A dictionary containing:
        - 'status': 'leaked' if leaking, else 'causally_sound'
        - 'is_leaking': True if loss < expected_entropy - tolerance
        - 'empirical_loss': Measured cross-entropy loss
        - 'expected_entropy': Theoretical Shannon entropy floor
        - 'gap': empirical_loss - expected_entropy
    """
    model.eval()
    inp = inputs if inputs is not None else self._inputs
    tgt = targets if targets is not None else self._targets

    with torch.no_grad():
      outputs = model(inp)
      if hasattr(outputs, "logits"):
        logits = outputs.logits
      elif isinstance(outputs, (tuple, list)):
        logits = outputs[0]
      else:
        logits = outputs

      eps = 1e-12
      if logits.dim() == 3:
        valid_logits = logits[:, self._order - 1:, :]
        valid_targets = tgt[:, self._order - 1:]
        loss = torch.nn.functional.cross_entropy(valid_logits.reshape(-1, self._vocab_size),
                                                 valid_targets.reshape(-1)).item()
        expected_h = float((-torch.sum(
          self._target_probs[:, self._order - 1:, :] *
          torch.log(self._target_probs[:, self._order - 1:, :] + eps),
          dim=-1,
        )).mean().item())
      else:
        loss = torch.nn.functional.cross_entropy(logits, tgt[:, -1]).item()
        expected_h = float((-torch.sum(
          self._target_probs[:, -1, :] * torch.log(self._target_probs[:, -1, :] + eps),
          dim=-1,
        )).mean().item())

    is_leaking = loss < (expected_h - tolerance)
    return {
      "status": "leaked" if is_leaking else "causally_sound",
      "is_leaking": is_leaking,
      "empirical_loss": loss,
      "expected_entropy": expected_h,
      "gap": loss - expected_h,
    }
