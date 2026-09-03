# syndset

[![CI](https://github.com/davide/syndset/actions/workflows/ci.yml/badge.svg)](https://github.com/davide/syndset/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/syndset/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Fast sanity checks and synthetic evaluation batteries for novel PyTorch neural architectures.

---

## Why syndset?

When you design a new architectural primitive—whether it is a custom attention variant, a novel recurrent cell, a hybrid state-space model, or an experimental normalization layer—the worst thing you can do is immediately launch a training job on ImageNet, OpenWebText, or WikiText.

Real-world datasets are slow, noisy, and expensive. When loss fails to decrease or plateaus early, you often don't know why:
- Did gradients vanish in the bottom layers?
- Did the activations collapse to a 1D line (dimensional collapse)?
- Can the architecture even memorize a single batch of 16 items?
- Is there a broken residual connection or an unscaled dot-product blowing up in half-precision?
- Can it solve the simplest algorithmic primitives (associative recall, selective copying, parity)?

`syndset` gives you a **10-second unit test suite** for your models. Before spending GPU hours, run an audit on your CPU or single GPU to verify structural health and inductive bias.

---

## Installation

Install from PyPI:

```bash
pip install syndset
```

Or install in editable mode for local development:

```bash
git clone https://github.com/davide/syndset.git
cd syndset
pip install -e .
```

`syndset` has only one required dependency: **`torch>=2.0.0`**.

---

## Quickstart: Audit Your Model in 3 Lines

```python
import torch
import torch.nn as nn
import syndset as syn

# Your experimental architecture
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))

# Run a complete diagnostic battery
report = syn.audit_tabular(model, dim=32, num_classes=4)
report.print_summary()
```

Output:

```text
======================================================================
  SYNDSET ARCHITECTURE AUDIT REPORT: Sequential (Tabular Task)
  Overall Health: [PASS]
======================================================================
Check                     | Status   | Finding
----------------------------------------------------------------------
Gradient Flow             | [PASS]   | Norm: 0.814, Vanishing ratio: 6.84e-01, Zero-grad params: 0
Effective Rank            | [PASS]   | 2 layers monitored, 0 collapsed
Dead Units                | [PASS]   | 0 layers with >30% dead units
Numerical Stability       | [PASS]   | Max activation: 0.42, Half-precision diff: 0.12%
Overfit Capacity          | [PASS]   | Final loss: 0.0008 (in 18 steps)
======================================================================
```

---

## Diagnostic Probes

You can run individual diagnostics directly whenever you need targeted checks:

### 1. Gradient Flow (`syn.check_gradient_flow`)
Hooks into backpropagation on a single step to verify that gradients reach every parameter.

```python
results = syn.check_gradient_flow(model, sample_input, target=sample_target)
print("Status:", results["status"])
print("Vanishing ratio:", results["vanishing_ratio"])
print("Unused parameters:", results["zero_grad_params"])
```

**What it catches:**
- **Dead parameters:** Parameters that receive zero gradient (often caused by accidental `.detach()`, missing outputs, or wrong branches).
- **Vanishing gradients:** Flags when bottom layers receive orders of magnitude less signal than top layers (ratio $< 10^{-5}$).
- **Exploding gradients:** Catches unnormalized activations driving norms to infinity or NaNs.

### 2. Effective Rank & Dimensional Collapse (`syn.check_effective_rank`)
Computes the spectral entropy-based effective rank of hidden layers using singular value decomposition (Roy & Vetterli, 2007):
$$\text{erank}(A) = \exp\left(-\sum_i p_i \ln p_i\right), \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}$$

```python
rank_info = syn.check_effective_rank(model, sample_input)
print("Layer ranks:", rank_info["layer_ranks"])
print("Collapsed layers:", rank_info["collapsed_layers"])
```

**What it catches:**
- **Representation collapse:** Deep networks sometimes project all representations into a tiny 1D or 2D subspace, losing expressivity despite high parameter count.

### 3. Dead Units Probe (`syn.check_dead_units`)
Measures the variance of activations across batch samples.

```python
dead_info = syn.check_dead_units(model, sample_input)
print("Dead ratios per layer:", dead_info["layer_dead_fractions"])
```

**What it catches:**
- Units or channels that produce constant or zero values for every sample (the classic "dying ReLU" failure mode or dead attention heads).

### 4. Single-Batch Overfit Capacity (`syn.check_overfit_capacity`)
Performs 50–100 quick optimization steps on an isolated copy of the model weights on a small batch.

```python
cap_info = syn.check_overfit_capacity(model, sample_input, sample_target)
print("Did it converge to ~0 loss?", cap_info["converged"])
print("Steps taken:", cap_info["steps_to_converge"])
```

**What it catches:**
- If an architecture cannot drive loss to zero on 8 or 16 samples, something is fundamentally wrong with the loss contract, scale of initializations, or gradient flow.

### 5. Numerical & Mixed-Precision Stability (`syn.check_numerical_stability`)
Tests whether activations explode when running under half precision (`bfloat16`/`float16`) and detects NaNs or denormals.

```python
stab_info = syn.check_numerical_stability(model, sample_input)
print("NaN present?", stab_info["has_nan"])
print("Relative difference in bfloat16:", stab_info["half_precision_relative_error"])
```

---

## Synthetic Benchmark Datasets

`syndset` includes procedural datasets designed to test specific algorithmic and spatial capabilities without external downloads.

### LLM & Sequence Models (`syndset.domains.llm`)

| Dataset | What It Tests |
| :--- | :--- |
| `AssociativeRecallDataset` | Key-value retrieval across sequence horizons (`[k1, v1, k2, v2, ..., kq] -> vq`). Critical benchmark for attention vs SSMs. |
| `InductionDataset` | In-context pattern completion (`[A, B, ..., A] -> B`), testing induction head mechanics. |
| `SelectiveCopyDataset` | Ability to filter out noise/distractor tokens and copy only marked signals in order. |
| `CumulativeParityDataset` | Binary sequence tracking cumulative XOR over time to test discrete compositionality. |

```python
from syndset.domains.llm import AssociativeRecallDataset

# 1000 sequences, 8 key-value pairs per sequence, vocab size 64
data = AssociativeRecallDataset(num_samples=1000, num_pairs=8, vocab_size=64)
tokens, target = data[0]
```

### Vision & Spatial Models (`syndset.domains.vision`)

| Dataset | What It Tests |
| :--- | :--- |
| `SyntheticShapesDataset` | Multi-class geometric shapes (circles, squares, triangles, pluses) with variable scale, offset, and noise. |
| `TextureVsShapeDataset` | Pits low-frequency shape silhouettes against conflicting high-frequency stripe textures to measure inductive bias. |
| `SpatialInvarianceDataset` | Pairs of base and translated shapes to quantify coordinate equivariance/invariance. |

### Tabular & Dense Models (`syndset.domains.tabular`)

| Dataset | What It Tests |
| :--- | :--- |
| `ConcentricHyperspheresDataset` | Points in nested $D$-dimensional spherical shells to test non-linear decision boundary capacity. |
| `SparseXORDataset` | High-dimensional inputs where only $k$ sparse coordinates determine target parity, testing feature selection. |
| `IllConditionedRegressionDataset` | Covariance matrix condition number scaled up to $10^4+$, testing optimization conditioning. |

### Time Series Models (`syndset.domains.timeseries`)

| Dataset | What It Tests |
| :--- | :--- |
| `HarmonicSuperpositionDataset` | Multi-frequency Fourier sums testing whether models capture high-frequency modes (spectral bias). |
| `AutoregressiveLagDataset` | Non-linear time series with distant lag dependencies to verify long-term memory. |

---

## Domain Audit Shortcuts

Run full audits tailored to specific model types with a single call:

```python
import syndset as syn

# Audit an LLM / sequence model
syn.audit_llm(my_transformer, num_pairs=8, vocab_size=64).print_summary()

# Audit a computer vision architecture
syn.audit_vision(my_convnet, img_size=32, channels=1).print_summary()

# Audit a tabular MLP
syn.audit_tabular(my_mlp, dim=16, num_classes=3).print_summary()

# Audit a time series / recurrent model
syn.audit_timeseries(my_rnn, seq_len=32).print_summary()
```

---

## Development & Publishing

### Running Tests

Run the complete test suite:

```bash
pytest -v
```

### Building the PyPI Distribution

To build wheels and source distributions:

```bash
python3 -m pip install build twine
python3 -m build
twine check dist/*
```

To upload to PyPI:

```bash
twine upload dist/*
```

---

## Code Style & Philosophy

- **Zero Bloat:** Pure PyTorch runtime. No heavy dependencies.
- **Fast:** Every test runs in milliseconds to seconds.
- **Clean Python:** Strict 2-space indentation, Google-style docstrings, and clean interfaces.
- **No Typing Annotations:** Dynamic, readable Python without type hint noise.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
