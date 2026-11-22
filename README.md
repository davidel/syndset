# syndset

[![CI](https://github.com/davidel/syndset/actions/workflows/ci.yml/badge.svg)](https://github.com/davidel/syndset/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/badge/pypi-v0.3.0-blue.svg)](https://pypi.org/project/syndset/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Fast sanity checks, diagnostic batteries, and zero-dependency synthetic datasets for PyTorch neural architectures.

---

## Table of Contents

- [Why syndset?](#why-syndset)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [1. Audit an Architecture in 3 Lines](#1-audit-an-architecture-in-3-lines)
  - [2. Load Synthetic Datasets Across Domains](#2-load-synthetic-datasets-across-domains)
- [Universal Synthetic Dataset Facility](#universal-synthetic-dataset-facility)
  - [Unified Data Contract (`DataSample`)](#unified-data-contract-datasample)
  - [Modality Catalog](#modality-catalog)
- [Multi-Model Architecture Tournaments](#multi-model-architecture-tournaments)
- [Diagnostic Probes Reference](#diagnostic-probes-reference)
  - [Gradients & Optimization](#gradients--optimization)
  - [Representation & Geometry](#representation--geometry)
  - [Numerical & Hardware Health](#numerical--hardware-health)
  - [Axiomatic Attribution & Conditioning](#axiomatic-attribution--conditioning)
  - [Architectural Symmetries](#architectural-symmetries)
- [Domain Audit Shortcuts](#domain-audit-shortcuts)
- [Mathematical Foundations](#mathematical-foundations)
- [Development](#development)
- [License](#license)

---

## Why syndset?

When testing a new architecture—such as an experimental attention mechanism, recurrent cell, state-space layer, or normalization scheme—launching a multi-hour training job on real-world datasets is inefficient. When loss plateaus or diverges, diagnosing the root cause is difficult:

- Did gradients vanish in early layers?
- Did hidden activations collapse to a low-dimensional subspace?
- Can the model memorize a single batch of 16 samples?
- Does half-precision float cause numerical overflow or catastrophic cancellation?
- Can the architecture solve basic algorithmic primitives (associative recall, selective copying, parity)?

`syndset` acts as a **unit testing suite for neural networks**. In seconds on CPU or GPU, it verifies structural health, gradient propagation, and algorithmic inductive bias before committing to full training runs.

---

## Installation

Install from PyPI:

```bash
pip install syndset
```

Or install in editable mode for local development:

```bash
git clone https://github.com/davidel/syndset.git
cd syndset
pip install -e .
```

`syndset` requires only **`torch>=2.0.0`**. No external GUI, computer vision, or simulation libraries are required.

---

## Quickstart

### 1. Audit an Architecture in 3 Lines

```python
import torch.nn as nn
import syndset as syn

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
Numerical Stability       | [PASS]   | Max act: 0.42, Half diff: 0.12%
Overfit Capacity          | [PASS]   | Final loss: 0.0008 (in 18 steps)
======================================================================
```

### 2. Load Synthetic Datasets Across Domains

```python
import syndset as syn

# Multimodal CLIP image-caption pairs with attribute-swapped hard negatives
clip_ds = syn.load_dataset("multimodal/shapes", num_samples=500)
sample = clip_ds[0]
print(sample.image.shape)    # torch.Size([3, 32, 32])
print(sample.caption)        # "a red circle in the top-left and a green square in the bottom-right"
print(sample.attr_negative)   # "a green circle in the top-left and a green square in the bottom-right"

# Visual Odometry with SE(3) camera motion and 3D landmarks
vo_ds = syn.load_dataset("odometry/vo", num_samples=200)
print(vo_ds[0].points_curr.shape)   # (50, 2)
print(vo_ds[0].rel_rotation.shape)  # (3, 3) rotation matrix in SO(3)

# CartPole offline RL transitions (s, a, r, s', done) in pure PyTorch
rl_ds = syn.load_dataset("rl/cartpole", policy="expert", num_samples=1000)
state, action, reward, next_state, done = rl_ds[0]

# Wrap any dataset into a PyTorch DataLoader
loader = clip_ds.to_dataloader(batch_size=32, shuffle=True)
```

---

## Universal Synthetic Dataset Facility

`syndset` includes procedurally generated micro-datasets to benchmark and sanity-check pipelines across modern AI domains without downloading external assets.

### Unified Data Contract (`DataSample`)

All datasets return `DataSample` objects that support dictionary access, attribute access, and backward-compatible tuple unpacking:

```python
sample = dataset[0]

# 1. Attribute access
img = sample.image
lbl = sample.label

# 2. Dict-style access
img = sample["image"]

# 3. Tuple unpacking (backward compatible)
x, y = sample
```

### Modality Catalog

| Domain | Dataset Identifier | Return Fields | Description & Inductive Bias |
| :--- | :--- | :--- | :--- |
| **Multimodal** | `multimodal/shapes` | `image`, `input_ids`, `caption`, `attr_negative`, `pos_negative` | Procedural paired RGB scenes with attribute and spatial hard-negative captions. |
| | `multimodal/vqa` | `image`, `question_ids`, `answer_id`, `question`, `answer` | Visual question answering covering color, existence, and spatial relations. |
| **Odometry** | `odometry/vo` | `points_prev`, `points_curr`, `rel_rotation`, `rel_translation`, `depth` | 3D landmark points under pinhole camera projection and rigid $`\mathrm{SE}(3)`$ motion. |
| | `odometry/optical_flow` | `image_prev`, `image_curr`, `flow` | Frame pairs with analytical dense optical flow vector fields in $`\mathbb{R}^{2 \times H \times W}`$. |
| **RL** | `rl/cartpole` | `state`, `action`, `reward`, `next_state`, `done` | Vectorized Newtonian CartPole kinematics under random, suboptimal, or expert policies. |
| | `rl/gridworld` | `state`, `action`, `reward`, `next_state` | $`N \times N`$ discrete grid navigation and shortest-path planning. |
| **Generative** | `generative/score` | `point`, `score` | Gaussian mixture with exact analytical score field $`\nabla_x \log p(x)`$ for diffusion testing. |
| | `generative/flow` | `x_t`, `time`, `target_velocity` | Optimal transport flow matching with exact conditional velocity vectors. |
| | `generative/manifolds_2d`| `point` | 2D/3D non-linear manifolds: Swiss roll, Pinwheel, and Checkerboard. |
| **Graphs** | `graphs/csl` | `adjacency`, `node_features`, `label` | Circular Skip Link graphs to test 1-Weisfeiler-Lehman (1-WL) expressivity limits. |
| | `graphs/shortest_path` | `adjacency`, `source_dest`, `distance` | Random graphs with exact Bellman-Ford shortest-path distances. |
| | `graphs/se3_equivariance`| `coordinates`, `charges`, `energy` | 3D molecular point clouds with exact $`\mathrm{SE}(3)`$-invariant Coulomb energy. |
| **Neural Fields**| `neural_fields/sdf` | `point`, `distance`, `normal` | Analytical signed distance functions (sphere, box, torus) and surface normals. |
| | `neural_fields/nerf` | `ray_origin`, `ray_dir`, `color` | Analytical ray-sphere intersection rendering for volume marcher testing. |
| **Interpretability**| `interpretability/grokking`| `inputs`, `target` | Modular arithmetic $`(a + b) \pmod p`$ for studying grokking and Fourier representations. |
| | `interpretability/sae` | `activation`, `true_features` | $`M`$ sparse features projected into a $`d \ll M`$ bottleneck to evaluate Sparse Autoencoders. |
| **Audio** | `audio/chirp` | `waveform`, `initial_freq`, `chirp_rate` | 1D frequency-modulated chirps testing 1D convolutions and STFT autograd. |
| | `audio/sound_events` | `waveform`, `event_mask` | Polyphonic sound sequences with exact ground-truth active time masks. |
| **Physics** | `physics/heat_equation` | `coordinates`, `solution` | Spatio-temporal coordinate queries $`(x, t) \to u(x, t)`$ with analytical Fourier diffusion solution. |
| | `physics/burgers_equation`| `initial_state`, `future_state` | Non-linear 1D viscous Burgers' PDE wave propagation. |
| **Continual** | `continual/rotating_stream`| `features`, `label`, `task_id` | Sequential tasks with rotating feature distributions for catastrophic forgetting. |
| | `continual/shortcut` | `features`, `label` | Core features paired with a spurious feature (95% aligned in train, 5% in test). |
| **LLM** | `llm/needle` | `inputs`, `target`, `needle_depth` | Needle-in-a-Haystack key-value retrieval across controllable sequence depths. |
| | `llm/grammar` | `inputs`, `target` | Synthetic JSON / tool-call sequences with balanced delimiters. |
| | `llm/dag` | `inputs`, `target` | Multi-hop transitive reasoning across directed acyclic graphs. |
| | `llm/mqar` | `inputs`, `targets` | Multi-Query Associative Recall testing sub-quadratic KV memory retention. |
| | `llm/dyck` | `inputs`, `targets` | Dyck-$`k`$ context-free grammar parsing testing pushdown stack depth. |
| | `llm/markov` | `inputs`, `targets` | Stationary order-$`k`$ Markov language process with theoretical entropy floor. |
| **Vision** | `vision/shapes` | `image`, `label` | 8-class parametric shape rasterization (circle, square, triangle, cross, etc.). |
| | `vision/occluded_detection`| `image`, `boxes`, `labels`, `occlusion_rates` | Depth-ordered multi-object scenes with bounding boxes and occlusion ratios. |
| | `vision/keypoints` | `image`, `keypoints` | 2D articulated planar stick figures with 7 kinematic joint coordinates. |
| | `vision/texture_vs_shape`| `image`, `shape_label`, `texture_label` | Geirhos-style shape vs. high-frequency texture bias benchmark. |
| | `vision/spatial_invariance`| `base_image`, `shifted_image`, `label` | Translation pairs quantifying spatial equivariance and feature distortion. |
| **Tabular** | `tabular/hyperspheres` | `features`, `label` | Linearly inseparable concentric hyperspherical shells in $`\mathbb{R}^D`$. |
| | `tabular/two_spirals` | `features`, `label` | Classic interlocked dual Archimedean spirals for non-linear boundary tests. |
| | `tabular/ill_conditioned_regression`| `features`, `target` | Extreme feature multi-collinearity with condition number $`\kappa \ge 10^4`$. |
| **Time Series** | `timeseries/harmonics` | `inputs`, `targets` | Multi-frequency harmonic Fourier superposition testing spectral bias. |
| | `timeseries/ar_lag` | `inputs`, `targets` | Long-range historical lag dependencies testing temporal retention. |

---

## Multi-Model Architecture Tournaments

When selecting among architectural variants, `syndset.Tournament` benchmarks candidates on identical synthetic tasks, applies hard veto filters, computes multi-objective Pareto frontiers, and produces actionable verdicts: **🟢 PROMOTE**, **🟡 CONDITIONALLY VIABLE**, or **🔴 DROP**.

```python
import syndset as syn

candidates = {
  "MHA_Baseline": Transformer(heads=8, kv_heads=8),
  "GQA_Candidate": Transformer(heads=8, kv_heads=2),
  "Broken_NoNorm": TransformerNoNorm(heads=8),
}

# Run tournament on synthetic LLM associative recall
report = syn.tournament_llm(candidates, num_pairs=8, vocab_size=64, profile="balanced")
report.print_summary()
```

Output:

```text
============================================================================================
  SYNDSET ARCHITECTURE TOURNAMENT: LLM (Profile: balanced)
============================================================================================
Rank  | Model                     | Score   | Verdict              | Status
--------------------------------------------------------------------------------------------
1     | GQA_Candidate             | 0.884   | 🟢 PROMOTE           | [PARETO]
       -> Pareto-optimal candidate with clean diagnostic health. Ready for training.
2     | MHA_Baseline              | 0.761   | 🔴 DROP              | [DOMINATED]
       -> Pareto-dominated by another candidate with superior expressivity/efficiency trade-off.
3     | Broken_NoNorm             |  N/A    | 🔴 DROP              | [VETO]
       -> Disqualified by hard veto gates. Do not train.
       -> Fatal numerical instability: NaN or Inf activations observed.
============================================================================================
```

Tournament shortcuts:
- `syn.tournament_llm(candidates, ...)`: Evaluates associative recall, induction copying, and attention metrics.
- `syn.tournament_vision(candidates, ...)`: Evaluates shape classification, ERF dispersion, and activation kurtosis.
- `syn.tournament_tabular(candidates, ...)`: Evaluates concentric hyperspheres manifold separation and rank preservation.
- `syn.tournament_timeseries(candidates, ...)`: Evaluates multi-frequency harmonic superposition and temporal gradients.

---

## Diagnostic Probes Reference

Every diagnostic probe attaches non-invasively to any PyTorch `nn.Module`. Complete mathematical derivations and failure regimes are detailed in [Diagnostic Probes & Decision Mechanics](docs/theory_diagnostics.md).

### Gradients & Optimization

- **Gradient Flow (`syn.check_gradient_flow`)**:
  Performs a single backward step to verify gradient propagation across all parameters.
  *Detects:* Disconnected layers ($`\|\nabla_\theta \mathcal{L}\| = 0`$), exponential vanishing ($`R_{\text{vanish}} \lt 10^{-5}`$), and exploding gradients.

- **Gradient Signal-to-Noise Ratio (`syn.check_gradient_snr`)**:
  Measures per-parameter gradient variance across micro-batches.
  *Detects:* High gradient variance ($`\text{SNR} \lt 0.1`$) indicating batch instability or noisy loss landscapes.

- **Data-Free Synaptic Flow (`syn.check_synflow`)**:
  Evaluates path-integrated parameter gradient flow without training data.
  *Detects:* Parameter starvation, bottleneck disconnections, and dead architectural branches.

- **Single-Batch Overfit Capacity (`syn.check_overfit_capacity`)**:
  Performs 50–100 quick optimization steps on an isolated parameter clone.
  *Detects:* Broken residual connections, capacity starvation, or initialization mismatches that prevent memorizing even 16 samples.

### Representation & Geometry

- **Effective Rank & Dimensional Collapse (`syn.check_effective_rank`)**:
  Computes the spectral entropy of hidden layer activations via Singular Value Decomposition.
  *Detects:* Representation collapse where activations flatten into a 1D/2D subspace.

- **Dead Units Probe (`syn.check_dead_units`)**:
  Measures activation variance across batch samples.
  *Detects:* Saturated non-linearities and dead neurons receiving zero gradient across all samples.

- **Layer CKA Similarity (`syn.check_layer_cka`)**:
  Computes linear Centered Kernel Alignment between intermediate representations across layers.
  *Detects:* Architectural depth redundancy (consecutive CKA $`\gt 0.99`$) and abrupt representational disconnects (CKA $`\lt 0.03`$).

- **Effective Receptive Field (`syn.check_effective_receptive_field`)**:
  Calculates the Gaussian dispersion of input-to-output gradients in spatial networks.
  *Detects:* Constricted receptive fields failing to aggregate spatial context.

### Numerical & Hardware Health

- **Numerical & Mixed-Precision Stability (`syn.check_numerical_stability`)**:
  Compares FP32 forward output against FP16/BF16 relative Frobenius error and scans for NaNs/Infs.
  *Detects:* Unscaled dot products exceeding FP16 limits ($`65,504`$) and catastrophic cancellation in normalizations.

- **Weight Initialization Scale & Spectral Radius (`syn.check_initialization_scale`)**:
  Compares parameter variance against Kaiming/Xavier bounds and computes square matrix spectral radius $`\rho(W)`$.
  *Detects:* Exploding/vanishing activation scaling and divergent recurrent/residual dynamics ($`\rho(W) \gt 1`$).

- **Activation Kurtosis & Quantization Health (`syn.check_quantization_health`)**:
  Measures sample kurtosis of layer activations.
  *Detects:* Heavy-tailed outlier activations ($`\text{Kurt}(X) \gt 20`$) that degrade INT8/FP8 post-training quantization.

- **Hardware & Efficiency Profiling (`syn.profile_model_efficiency`)**:
  Estimates parameter counts, FLOPs/MACs, and execution latency.
  *Detects:* Memory bottlenecks and disproportionate compute allocation across modules.

### Axiomatic Attribution & Conditioning

- **Attribution Faithfulness & Causal SNR (`syn.check_attribution_faithfulness`)**:
  Computes integrated gradients along a linear path and compares attributions against analytical ground-truth support masks.
  *Detects:* Reliance on distractor noise, inverted feature selection, and path-integration completeness divergence.

- **Layer Conductance & Algorithmic Depth (`syn.check_layer_conductance`)**:
  Decomposes model predictions into per-layer activation conductance shares.
  *Detects:* Premature decision bottlenecks and bypassed layers (<1% fair-share conductance).

- **Context Attribution & Recency Decay (`syn.check_context_attribution_decay`)**:
  Evaluates token attribution across sequence positions for autoregressive models.
  *Detects:* Early context blindness and severe recency bias ($`R_{\text{recency}} \gt 95\%`$).

- **Loss Surface Curvature & Sharpness (`syn.check_curvature_sharpness`)**:
  Estimates the Hessian trace $`\text{Tr}(H)`$ using Hutchinson's randomized estimator.
  *Detects:* Sharp loss ravines prone to optimization instability and poor generalization.

- **Empirical NTK Conditioning (`syn.check_ntk_conditioning`)**:
  Computes the condition number $`\kappa(\Theta) = \lambda_{\max} / \lambda_{\min}`$ of the empirical Neural Tangent Kernel.
  *Detects:* Ill-conditioned optimization dynamics and training stiffness ($`\kappa \gt 10^5`$).

### Architectural Symmetries

- **Permutation Equivariance & Invariance (`syn.check_permutation_equivariance`)**:
  Verifies coordinate permutation symmetries along a specified tensor axis.
  *Detects:* Unintended positional dependencies in set architectures (DeepSets, Set Transformers) and GNNs.

---

## Domain Audit Shortcuts

Run domain-specific audit batteries with customized synthetic inputs and tailored loss functions:

```python
import syndset as syn

# Tabular model audit
report = syn.audit_tabular(mlp_model, dim=32, num_classes=4)

# Vision classification audit
report = syn.audit_vision(conv_model, img_size=32, channels=3, num_classes=8)

# Object detection audit
report = syn.audit_detection(detector_model, img_size=32, num_objects=3)

# Semantic segmentation audit
report = syn.audit_segmentation(unet_model, img_size=32, num_classes=8)

# LLM / sequence model audit
report = syn.audit_llm(transformer_model, seq_len=64, vocab_size=64)

# Time series forecasting audit
report = syn.audit_timeseries(tcn_model, seq_len=64)

# View aggregated scorecard
report.print_summary()
```

---

## Mathematical Foundations

Complete mathematical derivations, condition bounds, and proofs are documented in dedicated reference files:

- 🔬 **[Diagnostic Probes & Decision Mechanics](docs/theory_diagnostics.md)**: Derivations for spectral entropy effective rank, gradient conditioning, CKA Hilbert-Schmidt independence criteria, SynFlow conservation, NTK eigenspectra, and multi-objective TOPSIS Pareto ranking.
- 📊 **[Synthetic Benchmark Tasks & Inductive Biases](docs/theory_datasets.md)**: Mathematical formulations for sequence/spatial task generation, manifold geometries, Chomsky Dyck grammars, induction copying circuits, and causal support masks.

---

## Development

### Running Tests
```bash
pytest -v
```

### Code Formatting & Linting
The codebase adheres to **Google Python Style** with **2-space indentation**, formatted using `yapf`:

```bash
# Check formatting
yapf --diff --recursive syndset tests

# Apply formatting in-place
yapf --in-place --recursive syndset tests

# Run linter
ruff check .
```

### Building for Release
```bash
python3 -m pip install build twine
python3 -m build
twine check dist/*
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
