# syndset

[![CI](https://github.com/davidel/syndset/actions/workflows/ci.yml/badge.svg)](https://github.com/davidel/syndset/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/syndset/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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
git clone https://github.com/davidel/syndset.git
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
Numerical Stability       | [PASS]   | Max act: 0.42, Half diff: 0.12%
Overfit Capacity          | [PASS]   | Final loss: 0.0008 (in 18 steps)
======================================================================
```

---

## Multi-Model Architecture Tournaments & Decision Engine

When evaluating several candidate architectures in the same domain (e.g. LLM attention variants, vision backbones, tabular networks), `syndset.Tournament` benchmarks them on identical synthetic micro-tasks, applies hard veto filters, computes multi-objective Pareto frontiers, and produces actionable verdicts: **🟢 PROMOTE**, **🟡 CONDITIONALLY VIABLE**, or **🔴 DROP**.

```python
import torch.nn as nn
import syndset as syn

candidates = {
  "MHA_Baseline": MyTransformer(heads=8, kv_heads=8),
  "GQA_Candidate": MyTransformer(heads=8, kv_heads=2),
  "Broken_NoNorm": MyTransformerNoNorm(heads=8),
}

# Run tournament on synthetic LLM associative recall
report = syn.tournament_llm(candidates, num_pairs=8, vocab_size=64, profile="balanced")
report.print_summary()
```

Output Scorecard:

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

### Domain Tournament Shortcuts:
* `syn.tournament_llm(candidates, ...)`: Evaluates associative recall, induction copying circuits, and attention metrics.
* `syn.tournament_vision(candidates, ...)`: Evaluates shape classification, ERF dispersion, and activation kurtosis.
* `syn.tournament_tabular(candidates, ...)`: Evaluates concentric hyperspheres manifold separation and rank preservation.
* `syn.tournament_timeseries(candidates, ...)`: Evaluates multi-frequency harmonic superposition and temporal gradients.

---

## Diagnostic Probes Overview

Every diagnostic probe attaches non-invasively to any standard PyTorch `nn.Module`. For complete mathematical derivations, symbol definitions, and failure regimes, see the [Mathematical Foundations & Theoretical Mechanics](#mathematical-foundations--theoretical-mechanics) section below.

### 1. Gradient Flow (`syn.check_gradient_flow`)
Hooks into backpropagation on a single step to verify that gradients reach every parameter.

```python
results = syn.check_gradient_flow(model, sample_input, target=sample_target)
print("Status:", results["status"])
print("Vanishing ratio:", results["vanishing_ratio"])
print("Unused parameters:", results["zero_grad_params"])
```

* **Detects:** Disconnected parameters ($`\large \|\nabla_\theta \mathcal{L}\| = 0`$), exponential gradient vanishing ($`\large R_{\text{vanish}} \lt 10^{-5}`$), and explosive gradient growth.
* [Jump to Gradient Flow Math & Derivation &rarr;](docs/theory_diagnostics.md#2-gradient-flow--jacobian-conditioning)

### 2. Effective Rank & Dimensional Collapse (`syn.check_effective_rank`)
Computes the spectral entropy-based effective rank of hidden layers using Singular Value Decomposition:

```python
rank_info = syn.check_effective_rank(model, sample_input)
print("Layer ranks:", rank_info["layer_ranks"])
print("Collapsed layers:", rank_info["collapsed_layers"])
```

* **Detects:** Representation collapse (where high-dimensional representations flatten into a 1D/2D subspace) and attention anisotropy.
* [Jump to Effective Rank Math & SVD Formulation &rarr;](docs/theory_diagnostics.md#1-effective-rank--dimensional-collapse)

### 3. Dead Units Probe (`syn.check_dead_units`)
Measures the variance of activations across batch samples.

```python
dead_info = syn.check_dead_units(model, sample_input)
print("Dead ratios per layer:", dead_info["layer_dead_fractions"])
```

* **Detects:** Saturated non-linearities and the classic "dying ReLU" failure mode where neurons receive zero gradient across all samples.
* [Jump to Dead Units Formulation &rarr;](docs/theory_diagnostics.md#4-dead-units--activation-saturation)

### 4. Single-Batch Overfit Capacity (`syn.check_overfit_capacity`)
Performs 50–100 quick optimization steps on an isolated parameter clone on a small batch.

```python
cap_info = syn.check_overfit_capacity(model, sample_input, sample_target)
print("Did it converge to ~0 loss?", cap_info["converged"])
print("Steps taken:", cap_info["steps_to_converge"])
```

* **Detects:** Deficient capacity, initialization scale mismatches, and broken skip/residual pathways that prevent learning even 16 samples.

### 5. Numerical & Mixed-Precision Stability (`syn.check_numerical_stability`)
Compares FP32 forward execution against half-precision (`bfloat16`/`float16`) using the relative Frobenius norm and detects NaNs/Infs.

```python
stab_info = syn.check_numerical_stability(model, sample_input)
print("NaN present?", stab_info["has_nan"])
print("Relative difference in bfloat16:", stab_info["half_precision_relative_error"])
```

* **Detects:** Activations exceeding IEEE 754 FP16 limits ($`65,504`$) due to unscaled dot products, and catastrophic cancellation in normalization layers.
* [Jump to Mixed-Precision Math & IEEE 754 Details &rarr;](docs/theory_diagnostics.md#5-floating-point-dynamic-range--mixed-precision-perturbations)

### 6. Weight Initialization Scaling & Spectral Radius (`syn.check_initialization_scale`)
Examines raw parameter distributions without passing data, comparing variances against Kaiming/Xavier bounds and computing spectral radius $`\rho(W)`$.

```python
init_info = syn.check_initialization_scale(model)
print("Status:", init_info["status"])
print("Spectral radii of square weights:", init_info["spectral_radii"])
```

* **Detects:** Miscalibrated weight scales causing exploding/vanishing activations, and square recurrent/residual matrices with divergent spectral radius ($`\large \rho(W) \gt 1`$).
* [Jump to Initialization & Spectral Radius Math &rarr;](docs/theory_diagnostics.md#6-weight-initialization-scaling--spectral-radius)

### 7. Loss Curvature & Sharpness (`syn.check_curvature_sharpness`)
Estimates Hessian trace $`\text{Tr}(H)`$ and mean loss curvature in 5 fast backward passes using Hutchinson's randomized estimator.

```python
curv_info = syn.check_curvature_sharpness(model, sample_input, sample_target)
print("Hessian trace estimate:", curv_info["hessian_trace"])
print("Mean curvature (eigenvalue):", curv_info["mean_curvature"])
```

* **Detects:** Hyper-sharp loss ravines prone to optimization instability and poor generalization.
* [Jump to Curvature & Hutchinson Trace Math &rarr;](docs/theory_diagnostics.md#7-loss-surface-curvature--hutchinsons-hessian-trace-estimator)

### 8. Permutation Equivariance & Invariance (`syn.check_permutation_equivariance`)
Verifies whether set architectures (DeepSets, Set Transformers) or GNNs satisfy coordinate permutation symmetries along a given axis.

```python
perm_info = syn.check_permutation_equivariance(model, sample_input, perm_dim=1)
print("Equivariant?", perm_info["is_satisfied"])
print("Relative error:", perm_info["relative_difference"])
```

* **Detects:** Unintended positional dependencies or asymmetric operations that violate permutation symmetry.
* [Jump to Permutation Equivariance Math &rarr;](docs/theory_diagnostics.md#8-permutation-equivariance--invariance)

### 9. Causal Attribution Faithfulness & SNR (`syn.check_attribution_faithfulness`)
Computes path-integrated gradients along a linear baseline trajectory and benchmarks attribution concentration against the dataset's analytical ground-truth causal support mask $`M^{\ast} \in \{0, 1\}^D`$.

```python
faith_info = syn.check_attribution_faithfulness(model, dataset)
print("Attribution SNR (dB):", faith_info["attribution_snr_db"])
print("Causal mass ratio:", faith_info["causal_mass_ratio"])
print("Top-K causal precision:", faith_info["topk_precision"])
print("Completeness error:", faith_info["mean_completeness_error"])
```

* **Detects:** Inverted feature selection ($`\large \text{SNR}_{\text{attr}} \lt 0 \text{ dB}`$), reliance on distractor noise, and path-integration completeness divergence caused by non-smooth activations or broken autograd formulas.
* [Jump to Causal Attribution Math & Derivation &rarr;](docs/theory_diagnostics.md#10-axiomatic-attribution--causal-ground-truth-mechanics)

### 10. Layer Conductance & Algorithmic Depth (`syn.check_layer_conductance`)
Decomposes output predictions into intermediate layer activation contributions using path-integrated activation conductance.

```python
cond_info = syn.check_layer_conductance(model, sample_input)
print("Layer conductance shares:", cond_info["layer_conductances"])
print("Effective depth ratio:", cond_info["effective_depth_ratio"])
print("Has dead/bypassed layers?", cond_info["has_dead_layers"])
```

* **Detects:** Layer underutilization, dead residual branches (<1% fair-share conductance), and premature decision bottlenecks where downstream layers act as redundant passthroughs.
* [Jump to Layer Conductance Formulation &rarr;](docs/theory_diagnostics.md#layer-conductance--algorithmic-depth-decomposition)

### 11. Context Attribution & Recency Decay (`syn.check_context_attribution_decay`)
Evaluates token attribution profiles across temporal sequence positions $`t \in \{0, \dots, T-1\}`$ for autoregressive and sequence architectures.

```python
decay_info = syn.check_context_attribution_decay(model, sample_tokens)
print("Recency ratio:", decay_info["recency_ratio"])
print("Early context blindness?", decay_info["early_context_blindness"])
```

* **Detects:** Early context blindness (first 20% of tokens receive zero attribution) and severe recency bias ($`\large R_{\text{recency}} \gt 95\%`$).
* [Jump to Context Decay Math & Recency Ratio &rarr;](docs/theory_diagnostics.md#context-attribution-decay--recency-bias)

### 12. Layer CKA & Representation Similarity (`syn.check_layer_cka`)
Computes linear Centered Kernel Alignment (CKA) between intermediate representations across all hidden layers:

```python
cka_res = syn.check_layer_cka(model, sample_input)
print("Consecutive similarities:", cka_res["consecutive_similarities"])
print("Redundant layers:", cka_res["redundant_layers"])
```

* **Detects:** Wasted architectural depth (consecutive CKA $`\gt 0.99`$) and sudden representational disconnects (CKA $`\lt 0.03`$).
* [Jump to Linear CKA Formulation &rarr;](docs/theory_diagnostics.md#11-centered-kernel-alignment-cka--representation-dynamics)

### 13. Data-Free Synaptic Flow (`syn.check_synflow`)
Evaluates synaptic flow conservation without training data using path-integrated absolute parameter gradients:

```python
syn_res = syn.check_synflow(model, sample_input)
print("Total synflow:", syn_res["total_synflow"])
print("Starved layers:", syn_res["starved_layers"])
```

* **Detects:** Unreached residual branches, parameter starvation, and completely disconnected computational paths.
* [Jump to SynFlow Conservation Math &rarr;](docs/theory_diagnostics.md#12-data-free-synaptic-flow-conservation-synflow)

### 14. Activation Kurtosis & Quantization Health (`syn.check_quantization_health`)
Evaluates sample kurtosis and extreme activation outliers ($`\gt 6\sigma`$) for FP8 and INT8 Post-Training Quantization (PTQ):

```python
quant_res = syn.check_quantization_health(model, sample_input)
print("Max kurtosis:", quant_res["max_kurtosis"])
print("Problematic layers:", quant_res["problematic_layers"])
```

* **Detects:** Heavy activation tails (kurtosis $`\gt 30.0`$) that collapse uniform quantization grids.
* [Jump to Kurtosis & Quantization Derivation &rarr;](docs/theory_diagnostics.md#13-activation-kurtosis--low-precision-ptq-viability)

### 15. Empirical NTK Conditioning (`syn.check_ntk_conditioning`)
Calculates the condition number of the empirical Neural Tangent Kernel on small sample batches:

```python
ntk_res = syn.check_ntk_conditioning(model, sample_input)
print("NTK Condition number:", ntk_res["condition_number"])
print("NTK Trace:", ntk_res["trace"])
```

* **Detects:** Optimization stiffness ($`\kappa \gt 10^5`$) and extreme convergence disparities across error modes.
* [Jump to NTK Conditioning Derivation &rarr;](docs/theory_diagnostics.md#14-empirical-neural-tangent-kernel-ntk-conditioning)

### 16. Effective Receptive Field (`syn.check_effective_receptive_field`)
Evaluates 2D Gaussian spatial dispersion and high-frequency checkerboard grid noise for vision models:

```python
erf_res = syn.check_effective_receptive_field(model, input_shape=(1, 3, 32, 32))
print("ERF Radius ratio:", erf_res["erf_radius_ratio"])
print("Checkerboard ratio:", erf_res["checkerboard_ratio"])
```

* **Detects:** Constricted spatial context reception and strided convolution aliasing artifacts.
* [Jump to ERF Dispersion Math &rarr;](docs/theory_diagnostics.md#15-effective-receptive-field-erf-spatial-dispersion)

### 17. Hardware & Systems Profiling (`syn.profile_model_efficiency`)
Measures parameter counts, estimated Multiply-Accumulate operations (MACs), and peak forward activation memory:

```python
prof = syn.profile_model_efficiency(model, sample_input)
print("Trainable parameters:", prof["trainable_params"])
print("Estimated MACs:", prof["estimated_macs"])
print("Activation memory MB:", prof["activation_memory_mb"])
```

* **Detects:** High activation memory footprints requiring gradient checkpointing and memory-bound layer configurations.

---

## Architecture Audit APIs & Domain Shortcuts

`syndset` provides a unified entrypoint `syn.audit()` along with domain-specific shortcuts (`audit_llm`, `audit_vision`, `audit_segmentation`, `audit_detection`, `audit_tabular`, `audit_timeseries`). In seconds, these run a battery of non-invasive diagnostics (gradient flow, effective rank, dead units, half-precision stability, and single-batch capacity) and return an `AuditReport` scorecard.

---

### Core Audit Runner: `syn.audit()`

```python
syn.audit(
  model,
  sample_input,
  sample_target=None,
  loss_fn=None,
  run_overfit=True,
  run_rank=True,
  run_stability=True,
  run_conductance=False,
  model_name=None,
)
```

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `torch.nn.Module` | *required* | The PyTorch architecture to audit. |
| `sample_input` | `Tensor` \| `tuple` \| `dict` | *required* | Sample batch matching `model.forward()`. Must include a batch dimension (e.g. `(B, ...)`). |
| `sample_target` | `Tensor` | `None` | Ground-truth target tensor matching `sample_input`. Required for overfit capacity probe when `run_overfit=True`. |
| `loss_fn` | `callable` | `None` | Custom loss `loss_fn(outputs, targets)`. If `None`, defaults to cross-entropy for 1D integer targets or MSE loss otherwise. |
| `run_overfit` | `bool` | `True` | Whether to run the single-batch memorization/overfit capacity probe. |
| `run_rank` | `bool` | `True` | Whether to compute spectral entropy effective rank and dead units on intermediate representations. |
| `run_stability` | `bool` | `True` | Whether to test numerical stability under FP16/BF16 half-precision and detect NaN/inf gradients. |
| `run_conductance` | `bool` | `False` | Whether to compute layer conductance flow and detect dead/bypassed intermediate blocks. |
| `model_name` | `str` | `None` | Identifier string for scorecard header (defaults to `model.__class__.__name__`). |

#### Return Value: `AuditReport`

`syn.audit()` returns an `AuditReport` instance aggregating findings across all probes:

- `report.is_healthy()` &rarr; `bool`: Returns `True` if no checks failed (`PASS` or `WARN`).
- `report.overall_status` &rarr; `str`: Aggregate rating (`"passed"`, `"warning"`, or `"failed"`).
- `report.print_summary()`: Prints the ASCII scorecard directly to stdout.
- `report.summary()` &rarr; `str`: Returns the multi-line formatted scorecard string.
- `report.to_dict()` &rarr; `dict`: Structured dictionary representation containing status and probe findings.
- `report.checks` &rarr; `list[dict]`: Individual check records (`name`, `tag`, `summary`, `details`).

```python
import torch
import torch.nn as nn
import syndset as syn

model = nn.Sequential(
  nn.Linear(32, 64),
  nn.GELU(),
  nn.Linear(64, 4),
)

inputs = torch.randn(16, 32)
targets = torch.randint(0, 4, (16,))

# Run complete audit
report = syn.audit(model, inputs, sample_target=targets, model_name="FeedForwardNet")
report.print_summary()

if not report.is_healthy():
  print("Audit failed! Issues:", report.to_dict()["checks"])
```

---

### Domain-Specific Audit Shortcuts

Domain shortcuts automatically instantiate the corresponding synthetic benchmark dataset, draw a batch of inputs and targets, and execute the audit battery with zero boilerplate.

| Function | Domain | Synthetic Dataset Used | Expected Model Inputs | Expected Model Outputs |
| :--- | :--- | :--- | :--- | :--- |
| [`syn.audit_llm`](#1-audit_llm) | Sequence / Autoregressive | `AssociativeRecallDataset` | `(B, 2*num_pairs + 1)` int64 token IDs | Logits `(B, vocab_size)` or `(B, seq_len, vocab_size)` |
| [`syn.audit_vision`](#2-audit_vision) | Image Classification | `SyntheticShapesDataset` | `(B, C, H, W)` float32 in `[0, 1]` | Class logits `(B, num_classes)` |
| [`syn.audit_segmentation`](#3-audit_segmentation) | Dense Semantic Segmentation | `SyntheticSegmentationDataset` | `(B, C, H, W)` float32 in `[0, 1]` | Dense logits `(B, num_classes + 1, H, W)` |
| [`syn.audit_detection`](#4-audit_detection) | Object Detection | `SyntheticDetectionDataset` | `(B, C, H, W)` float32 in `[0, 1]` | Box coordinates `(B, num_objects, 4)` in `[0, 1]` |
| [`syn.audit_tabular`](#5-audit_tabular) | Tabular / Feedforward | `ConcentricHyperspheresDataset` | `(B, dim)` float32 | Class logits `(B, num_classes)` |
| [`syn.audit_timeseries`](#6-audit_timeseries) | Time Series / Recurrent | `HarmonicSuperpositionDataset` | `(B, seq_len - 1, 1)` float32 | Step forecast `(B, seq_len - 1, 1)` |

#### 1. `audit_llm`

Audits an autoregressive transformer, SSM (Mamba, S4), or recurrent sequence model on key-value associative retrieval.

```python
syn.audit_llm(model, num_pairs=8, vocab_size=64, batch_size=8, seed=42)
```

- **Parameters**:
  - `model`: Sequence `torch.nn.Module`. Must accept token ID tensors of shape `(batch_size, 2 * num_pairs + 1)` with dtype `torch.long`.
  - `num_pairs` (`int`, default: `8`): Number of distinct (key, value) pairs. Total sequence length is `2 * num_pairs + 1`.
  - `vocab_size` (`int`, default: `64`): Vocabulary size. Must satisfy `vocab_size >= 2 * num_pairs`.
  - `batch_size` (`int`, default: `8`): Batch size for audit inputs.
  - `seed` (`int`, default: `42`): Random seed.

```python
import torch.nn as nn
import syndset as syn

class MiniTransformer(nn.Module):
  def __init__(self, vocab_size=64, d_model=32):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, d_model)
    layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=64, batch_first=True)
    self.encoder = nn.TransformerEncoder(layer, num_layers=2)
    self.head = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    h = self.encoder(self.emb(x))
    return self.head(h[:, -1, :])  # Predict from last token

model = MiniTransformer(vocab_size=64)
syn.audit_llm(model, num_pairs=8, vocab_size=64).print_summary()
```

#### 2. `audit_vision`

Audits a vision classifier (CNN, ViT, MLP-Mixer) on multi-class procedural geometric shapes (8 classes: circle, square, triangle, cross, rectangle, ellipse, annulus, diamond).

```python
syn.audit_vision(model, img_size=32, channels=1, num_classes=None, batch_size=8, seed=42)
```

- **Parameters**:
  - `model`: Vision `torch.nn.Module`. Accepts `(batch_size, channels, img_size, img_size)` float32 tensors and returns `(batch_size, num_classes)` logits.
  - `img_size` (`int`, default: `32`): Image height and width in pixels.
  - `channels` (`int`, default: `1`): Number of image color channels.
  - `num_classes` (`int`, default: `None`): Number of classes in `[1, 8]`. If `None`, inferred from `model.num_classes` or `model.classifier.out_features` (defaults to 8).
  - `batch_size` (`int`, default: `8`): Batch size for audit.
  - `seed` (`int`, default: `42`): Random seed.

```python
import torch.nn as nn
import syndset as syn

model = nn.Sequential(
  nn.Conv2d(1, 16, kernel_size=3, padding=1),
  nn.BatchNorm2d(16),
  nn.ReLU(),
  nn.AdaptiveAvgPool2d((1, 1)),
  nn.Flatten(),
  nn.Linear(16, 8),
)

syn.audit_vision(model, img_size=32, channels=1, num_classes=8).print_summary()
```

#### 3. `audit_segmentation`

Audits a dense semantic segmentation architecture (U-Net, FPN, DeepLab, SegFormer) on procedural multi-object pixel masks using `nn.CrossEntropyLoss`. Evaluates encoder-decoder skip connection gradients and bottleneck representation rank.

```python
syn.audit_segmentation(model, img_size=32, channels=1, num_classes=None, num_objects=3, batch_size=8, seed=42)
```

- **Parameters**:
  - `model`: Dense segmentation `torch.nn.Module`. Must accept `(batch_size, channels, img_size, img_size)` and output `(batch_size, num_classes + 1, img_size, img_size)` logits.
  - `img_size` (`int`, default: `32`): Image resolution in pixels.
  - `channels` (`int`, default: `1`): Number of color channels.
  - `num_classes` (`int`, default: `None`): Number of foreground shape classes in `[1, 8]`. Total classes will be `num_classes + 1` (class 0 is background). If `None`, inferred from `model.out_channels - 1` (defaults to 8).
  - `num_objects` (`int`, default: `3`): Number of geometric shapes placed on each canvas.
  - `batch_size` (`int`, default: `8`): Batch size for audit.
  - `seed` (`int`, default: `42`): Random seed.

```python
import torch
import torch.nn as nn
import syndset as syn

class ToyUNet(nn.Module):
  def __init__(self, in_channels=1, num_classes=9):
    super().__init__()
    self.enc = nn.Conv2d(in_channels, 16, 3, padding=1)
    self.dec = nn.Conv2d(16, num_classes, 3, padding=1)

  def forward(self, x):
    h = torch.relu(self.enc(x))
    return self.dec(h)

model = ToyUNet(in_channels=1, num_classes=9)
syn.audit_segmentation(model, img_size=32, channels=1, num_classes=8).print_summary()
```

#### 4. `audit_detection`

Audits an object detector or coordinate regression head (DETR, YOLO, RetinaNet) on procedural bounding box targets. Evaluates coordinate regression pathways and anchor assignment gradients using Smooth L1 loss.

```python
syn.audit_detection(model, img_size=32, channels=1, num_objects=3, num_classes=8, batch_size=8, loss_fn=None, seed=42)
```

- **Parameters**:
  - `model`: Detector `torch.nn.Module`. Accepts `(batch_size, channels, img_size, img_size)` and must output predicted bounding boxes of shape `(batch_size, num_objects, 4)` in normalized `[cx, cy, w, h]` format in `[0.0, 1.0]` (or dict/tuple containing box predictions).
  - `img_size` (`int`, default: `32`): Image resolution in pixels.
  - `channels` (`int`, default: `1`): Number of color channels.
  - `num_objects` (`int`, default: `3`): Number of objects per image.
  - `num_classes` (`int`, default: `8`): Geometric shape classes.
  - `batch_size` (`int`, default: `8`): Batch size for audit.
  - `loss_fn` (`callable`, default: `None`): Optional loss function. If `None`, defaults to Smooth L1 loss on box coordinates.
  - `seed` (`int`, default: `42`): Random seed.

```python
import torch
import torch.nn as nn
import syndset as syn

class ToyDetector(nn.Module):
  def __init__(self, num_objects=3):
    super().__init__()
    self.num_objects = num_objects
    self.backbone = nn.Sequential(
      nn.Conv2d(1, 16, 3, padding=1),
      nn.ReLU(),
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
    )
    self.box_head = nn.Linear(16, num_objects * 4)

  def forward(self, x):
    h = self.backbone(x)
    return torch.sigmoid(self.box_head(h)).view(-1, self.num_objects, 4)

detector = ToyDetector(num_objects=3)
syn.audit_detection(detector, img_size=32, channels=1, num_objects=3).print_summary()
```

#### 5. `audit_tabular`

Audits a feedforward architecture or tabular MLP on concentric hyperspherical shells. Evaluates whether the architecture possesses sufficient non-linear depth and capacity to separate non-linearly separable topological shells.

```python
syn.audit_tabular(model, dim=16, num_classes=3, batch_size=16, seed=42)
```

- **Parameters**:
  - `model`: Tabular `torch.nn.Module`. Accepts `(batch_size, dim)` float32 feature tensors and outputs `(batch_size, num_classes)` logits.
  - `dim` (`int`, default: `16`): Dimensionality of input feature vectors.
  - `num_classes` (`int`, default: `3`): Number of concentric spherical shells.
  - `batch_size` (`int`, default: `16`): Batch size for audit.
  - `seed` (`int`, default: `42`): Random seed.

```python
import torch.nn as nn
import syndset as syn

model = nn.Sequential(
  nn.Linear(16, 32),
  nn.ReLU(),
  nn.Linear(32, 32),
  nn.ReLU(),
  nn.Linear(32, 3),
)

syn.audit_tabular(model, dim=16, num_classes=3).print_summary()
```

#### 6. `audit_timeseries`

Audits a recurrent network (LSTM, GRU), causal convolution (TCN), or state-space model on harmonic Fourier waveforms. Evaluates whether high-frequency spectral modes are preserved or dampened (spectral bias).

```python
syn.audit_timeseries(model, seq_len=32, batch_size=8, seed=42)
```

- **Parameters**:
  - `model`: Time-series `torch.nn.Module`. Accepts `(batch_size, seq_len - 1, 1)` float32 sequence tensors and outputs forecasted values of matching shape `(batch_size, seq_len - 1, 1)`.
  - `seq_len` (`int`, default: `32`): Time steps in generated waveforms. Model input length is `seq_len - 1`.
  - `batch_size` (`int`, default: `8`): Batch size for audit.
  - `seed` (`int`, default: `42`): Random seed.

```python
import torch.nn as nn
import syndset as syn

class SimpleForecaster(nn.Module):
  def __init__(self):
    super().__init__()
    self.rnn = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
    self.head = nn.Linear(16, 1)

  def forward(self, x):
    h, _ = self.rnn(x)
    return self.head(h)

model = SimpleForecaster()
syn.audit_timeseries(model, seq_len=32).print_summary()
```

#### 7. `audit_attribution`

Audits causal feature attribution faithfulness and path completeness against a dataset's analytical causal support mask (`SparseXORDataset`, `AssociativeRecallDataset`, `SelectiveCopyDataset`, `TextureVsShapeDataset`). Evaluates Attribution SNR (dB), Causal Mass Ratio, Top-K Precision, and path completeness error in seconds.

```python
syn.audit_attribution(model, dataset, sample_indices=None, num_steps=20, snr_threshold_db=3.0)
```

- **Parameters**:
  - `model`: PyTorch `torch.nn.Module`.
  - `dataset`: Any `SyntheticDataset` implementing `.causal_mask(idx)`.
  - `sample_indices` (`list[int]`, default: first `8` samples): Indices of benchmark samples to evaluate.
  - `num_steps` (`int`, default: `20`): Number of Riemann interpolation steps along path.
  - `snr_threshold_db` (`float`, default: `3.0`): Minimum acceptable Attribution SNR in decibels.
  - `embedding_module` (`nn.Module`, optional): Explicit `nn.Embedding` submodule for discrete sequence models.

```python
import torch.nn as nn
import syndset as syn

dataset = syn.SparseXORDataset(num_samples=32, total_dim=32, active_dim=4)
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))

syn.audit_attribution(model, dataset).print_summary()
```

---

## Synthetic Benchmark Datasets Overview

`syndset` includes 20 procedural dataset generators across 4 domains. All datasets inherit from `syndset.domains.base.SyntheticDataset` (a standard PyTorch `torch.utils.data.Dataset`), providing:

- **Zero downloads**: Generated procedurally in CPU memory via PyTorch generators in milliseconds.
- **Deterministic reproducibility**: Controlled random seed management (`seed=42`).
- **Standard PyTorch compatibility**: Works directly with `torch.utils.data.DataLoader`, `dataset[idx]`, and slicing `dataset[:batch_size]`.
- **Full precomputed attributes**: Access entire generated arrays via properties (`.inputs`, `.targets`, `.images`, `.labels`, `.data`, etc.).

---

### LLM & Sequence Models (`syndset.domains.llm`)

Evaluates key-value associative retrieval, induction heads, noise filtering, stack memory, and causal next-token distributions.

| Dataset | Evaluates | `inputs` Shape & Dtype | `targets` Shape & Dtype | `__getitem__(idx)` Return Format |
| :--- | :--- | :--- | :--- | :--- |
| `AssociativeRecallDataset` | Key-value retrieval | `(N, 2*num_pairs + 1)` int64 | `(N,)` int64 | `(input_seq, target_val)` |
| `MultiQueryAssociativeRecallDataset` | Multi-Query Recall (MQAR) | `(N, 2*num_pairs + num_queries)` int64 | `(N, num_queries)` int64 | `(input_seq, target_vals)` |
| `InductionDataset` | In-context induction heads | `(N, seq_len)` int64 | `(N,)` int64 | `(tokens, target_token)` |
| `SelectiveCopyDataset` | Noise filtering & memory order | `(N, total_len)` int64 | `(N, num_signals)` int64 | `(noisy_seq, signal_seq)` |
| `CumulativeParityDataset` | Step-by-step XOR composition | `(N, seq_len)` float32 | `(N, seq_len)` int64 | `(bits, parity_targets)` |
| `DyckLanguageDataset` | Stack memory (Dyck-k) | `(N, seq_len)` int64 | `(N, seq_len)` int64 | `(seq, stack_targets)` |
| `MarkovLanguageDataset` | Next-token distribution modeling | `(N, seq_len)` int64 | `(N, seq_len)` int64 | `(input_seq, target_seq)` |

#### Detailed Dataset Breakdown & Generation Mechanics

##### `AssociativeRecallDataset`
- **Evaluates**: Key-value associative binding across sequence horizons (the core prerequisite for in-context retrieval in Transformers, Mamba, and RWKV).
- **What it generates**:
  1. *Vocabulary Partitioning*: Splits vocabulary into disjoint key pool `[1, vocab_size // 2]` and value pool `[vocab_size // 2 + 1, vocab_size - 1]`. Token `0` is reserved. Disjoint token sets eliminate role ambiguity.
  2. *Interleaved Key-Value Pairs*: Samples `num_pairs` distinct keys (without replacement) and `num_pairs` values (with replacement). Interleaves them into `[k_0, v_0, k_1, v_1, ..., k_{n-1}, v_{n-1}]`.
  3. *Query Key*: Uniformly picks one earlier key $`q = k_\tau`$ and appends it at the sequence end (total length: `2 * num_pairs + 1`).
  4. *Target*: Scalar ground-truth value token $`v_\tau`$ paired with $`k_\tau`$.
- **Sequence Layout**:
  - `inputs`: `[ k_0, v_0, k_1, v_1, ..., k_{n-1}, v_{n-1}, query_key ]` (shape: `(N, 2*num_pairs + 1)`, dtype: `int64`)
  - `targets`: `query_value` (shape: `(N,)`, dtype: `int64`)
- **Concrete Example**:
  With `num_pairs=3, vocab_size=16` (keys in `1..8`, values in `9..15`):
  Pairs generated: `(3 -> 11), (7 -> 14), (2 -> 10)`. Query chosen: key `7`.
  - `inputs[0]  = [ 3, 11, 7, 14, 2, 10, 7 ]`
  - `targets[0] = 14`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `num_pairs` (`int`): Key-value pairs per sequence (default: `8`). Sequence length is `2 * num_pairs + 1`.
  - `vocab_size` (`int`): Total vocabulary size (default: `64`). Must satisfy `vocab_size >= 2 * num_pairs`.
  - `seed` (`int`): Random seed (default: `42`).

##### `MultiQueryAssociativeRecallDataset`
- **Evaluates**: Multi-Query Associative Recall (MQAR), probing the memory capacity trade-off between standard softmax attention and fixed-state recurrent/SSM hidden representations when retrieving multiple distinct queries.
- **What it generates**:
  1. *Interleaved Prefix*: Constructs `num_pairs` interleaved key-value pairs `[k_0, v_0, ..., k_{n-1}, v_{n-1}]`.
  2. *Multi-Query Suffix*: Samples `num_queries` distinct query keys from $`\{k_0, \dots, k_{n-1}\}`$ without replacement and appends them at prompt end.
  3. *Target*: 1D tensor of `num_queries` values corresponding to the queried keys.
- **Sequence Layout**:
  - `inputs`: `[ k_0, v_0, ..., k_{n-1}, v_{n-1}, q_0, ..., q_{m-1} ]` (shape: `(N, 2*num_pairs + num_queries)`, dtype: `int64`)
  - `targets`: `[ v_{q_0}, v_{q_1}, ..., v_{q_{m-1}} ]` (shape: `(N, num_queries)`, dtype: `int64`)
- **Concrete Example**:
  With `num_pairs=4, num_queries=2, vocab_size=32`:
  Pairs: `(5 -> 20), (2 -> 29), (11 -> 18), (8 -> 24)`. Queries selected: `[11, 5]`.
  - `inputs[0]  = [ 5, 20, 2, 29, 11, 18, 8, 24, 11, 5 ]`
  - `targets[0] = [ 18, 20 ]`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `num_pairs` (`int`): Key-value pairs per sequence (default: `8`).
  - `num_queries` (`int`): Distinct query keys at sequence end (default: `3`, must satisfy `num_queries <= num_pairs`).
  - `vocab_size` (`int`): Total vocabulary size (default: `64`, must satisfy `vocab_size >= 2 * num_pairs`).
  - `seed` (`int`): Random seed (default: `42`).

##### `InductionDataset`
- **Evaluates**: In-context induction heads ($`A \dots B \dots A \to B`$), the fundamental 2-layer attention subcircuit responsible for in-context pattern replication in LLMs.
- **What it generates**:
  1. *Prefix Trigger*: Tokens `A = tokens[0]` and `B = tokens[1]` establish an initial trigger transition $`A \to B`$.
  2. *Distractor Horizon*: Random tokens fill intermediate positions `tokens[2 : seq_len - 1]`.
  3. *Trigger Re-injection*: Token `A` is re-inserted at the final prompt position: `tokens[-1] = A`.
  4. *Target*: Scalar continuation token `B`.
- **Sequence Layout**:
  - `inputs`: `[ A, B, r_2, r_3, ..., r_{T-2}, A ]` (shape: `(N, seq_len)`, dtype: `int64`)
  - `targets`: `B` (shape: `(N,)`, dtype: `int64`)
- **Concrete Example**:
  With `seq_len=8, vocab_size=32`:
  Trigger pattern: `A = 14, B = 27`.
  - `inputs[0]  = [ 14, 27, 3, 19, 8, 22, 11, 14 ]`
  - `targets[0] = 27`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `seq_len` (`int`): Total sequence length (default: `32`).
  - `vocab_size` (`int`): Vocabulary size (default: `64`).
  - `seed` (`int`): Random seed (default: `42`).

##### `SelectiveCopyDataset`
- **Evaluates**: Memory filtering and noise suppression, testing whether an architecture can ignore filler tokens and extract signal tokens in exact sequential order.
- **What it generates**:
  1. *Distractor Masking*: Sequence of length `total_len` initialized with distractor token `0`.
  2. *Signal Placement*: Samples `num_signals` sorted positions in the first half `[0, total_len // 2)` and inserts random non-zero tokens in `[1, vocab_size - 1]`.
  3. *Target*: 1D tensor of the `num_signals` signal tokens in order of occurrence.
- **Sequence Layout**:
  - `inputs`: `[ 0, ..., s_0, 0, ..., s_1, 0, ..., 0, 0 ]` (shape: `(N, total_len)`, dtype: `int64`)
  - `targets`: `[ s_0, s_1, ..., s_{k-1} ]` (shape: `(N, num_signals)`, dtype: `int64`)
- **Concrete Example**:
  With `total_len=10, num_signals=3, vocab_size=16`:
  Signals `[11, 7, 14]` placed at positions `[1, 2, 4]`.
  - `inputs[0]  = [ 0, 11, 7, 0, 14, 0, 0, 0, 0, 0 ]`
  - `targets[0] = [ 11, 7, 14 ]`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `num_signals` (`int`): Signal tokens to remember (default: `4`).
  - `total_len` (`int`): Total sequence length (default: `32`).
  - `vocab_size` (`int`): Signal vocabulary size (default: `32`). Token `0` is noise.
  - `seed` (`int`): Random seed (default: `42`).

##### `CumulativeParityDataset`
- **Evaluates**: Step-by-step XOR compositionality and deep discrete state tracking over time.
- **What it generates**:
  1. *Binary Stream*: Samples `seq_len` binary bits $`b_t \in \{0, 1\}`$.
  2. *Step Parity*: Computes cumulative sum modulo 2 at each step: $`y_t = \left(\sum_{i=0}^t b_i\right) \bmod 2`$.
- **Sequence Layout**:
  - `inputs`: `[ b_0, b_1, ..., b_{T-1} ]` (shape: `(N, seq_len)`, dtype: `float32` in `{0.0, 1.0}`)
  - `targets`: `[ y_0, y_1, ..., y_{T-1} ]` (shape: `(N, seq_len)`, dtype: `int64` in `{0, 1}`)
- **Concrete Example**:
  With `seq_len=8`:
  - `inputs[0]  = [ 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0 ]`
  - `targets[0] = [ 1,   1,   0,   1,   1,   0,   0,   0   ]`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `seq_len` (`int`): Length of binary sequence (default: `32`).
  - `seed` (`int`): Random seed (default: `42`).

##### `DyckLanguageDataset`
- **Evaluates**: Hierarchical stack memory and context-free grammar parsing (Dyck-$`k`$). Probes pushdown automaton simulation.
- **What it generates**:
  1. *Bracket Tokens*: Encodes $`k`$ bracket types as open $`= 2t - 1`$, close $`= 2t`$ (e.g. `( = 1, ) = 2, [ = 3, ] = 4`).
  2. *Pushdown Transitions*: Simulates valid stack pushes/pops up to `max_depth`.
  3. *Step Closure Targets*: After each emitted token, `targets[t]` records the closing token matching the current top of stack (or `0` if empty).
- **Sequence Layout**:
  - `inputs`: `[ b_0, b_1, ..., b_{T-1} ]` (shape: `(N, seq_len)`, dtype: `int64`)
  - `targets`: `[ c_0, c_1, ..., c_{T-1} ]` (shape: `(N, seq_len)`, dtype: `int64`)
- **Concrete Example**:
  With `num_types=2, seq_len=6`:
  Emitted brackets: `( [ ] [ ] )`
  - `inputs[0]  = [ 1, 3, 4, 3, 4, 2 ]`
  - `targets[0] = [ 2, 4, 2, 4, 2, 0 ]`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `seq_len` (`int`): Length of bracket sequence (default: `32`).
  - `num_types` (`int`): Number of bracket pairs, e.g. 2 for `()`, `[]` (default: `2`).
  - `max_depth` (`int`): Maximum stack depth allowed (default: `6`).
  - `seed` (`int`): Random seed (default: `42`).

##### `MarkovLanguageDataset`
- **Evaluates**: Statistical next-token distribution modeling, calibration, and causal validity against a known Bayes-optimal entropy floor.
- **What it generates**:
  1. *Dirichlet Transition Tensor*: Samples transition distributions $`P(x_t \mid x_{t-k:t-1}) \sim \text{Dirichlet}(\alpha \mathbf{1}_V)`$ of shape `(vocab_size**order, vocab_size)`.
  2. *Autoregressive Trajectory*: Generates sequence by sampling sequentially from the ground-truth Markov chain.
  3. *Shifted Prediction*: Input is `tokens[:-1]`, target is shifted `tokens[1:]`.
  4. *Analytical Attributes*: Stores true conditional probability distributions in `.target_probs` of shape `(N, seq_len, vocab_size)` and exact Shannon entropy floor in `.theoretical_entropy`.
- **Sequence Layout**:
  - `inputs`: `[ x_0, x_1, ..., x_{T-1} ]` (shape: `(N, seq_len)`, dtype: `int64`)
  - `targets`: `[ x_1, x_2, ..., x_T ]` (shape: `(N, seq_len)`, dtype: `int64`)
  - `target_probs`: True next-token probability distributions (shape: `(N, seq_len, vocab_size)`, dtype: `float32`)
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `seq_len` (`int`): Sequence length (default: `32`).
  - `vocab_size` (`int`): Discrete vocabulary size $`V \ge 2`$ (default: `8`).
  - `order` (`int`): Markov context order $`k \ge 1`$ (default: `2`).
  - `alpha` (`float`): Dirichlet concentration parameter for transition distribution prior (default: `1.0`).
  - `seed` (`int`): Random seed (default: `42`).
  - **Special Attributes & Methods**:
    - `.theoretical_entropy`: Bayes-optimal Shannon entropy in nats.
    - `.target_probs`: Tensor of shape `(num_samples, seq_len, vocab_size)` with true next-token probability distributions.
    - `.transition_matrix`: Full transition matrix of shape `(vocab_size**order, vocab_size)`.
    - `.prefix_grid`: Tuple of `(prefixes, prefix_distributions)` testing all $`V^k`$ context combinations in a single batch.
    - `.evaluate_distribution(model)`: Computes Total Variation (TV) distance and KL divergence against true Markov distributions.
    - `.check_causal_leakage(model)`: Detects if model loss falls below theoretical entropy (indicating future token lookahead).

```python
from torch.utils.data import DataLoader
from syndset.domains.llm import (
  AssociativeRecallDataset,
  MarkovLanguageDataset,
  MultiQueryAssociativeRecallDataset,
)

# 1. Multi-Query Associative Recall (MQAR)
mqar = MultiQueryAssociativeRecallDataset(num_samples=500, num_pairs=8, num_queries=3, vocab_size=64)
seq, query_targets = mqar[0]
print(f"MQAR sequence length: {seq.shape[0]}, target queries: {query_targets.shape[0]}")

# 2. Markov Language Dataset with theoretical entropy floor
markov = MarkovLanguageDataset(num_samples=500, seq_len=32, vocab_size=8, order=2)
print(f"Theoretical Bayes entropy floor: {markov.theoretical_entropy:.3f} nats")
print(f"Prefix grid shape: {markov.prefix_grid[0].shape}")

# Use with PyTorch DataLoader
loader = DataLoader(markov, batch_size=32, shuffle=True)
for batch_inputs, batch_targets in loader:
  # batch_inputs: (32, 32), batch_targets: (32, 32)
  break
```

---

### Vision & Spatial Models (`syndset.domains.vision`)

Evaluates spatial inductive biases, semantic segmentation, coordinate regression, numerosity counting, texture vs. shape bias, and translation invariance.

| Dataset | Evaluates | `images` Shape & Dtype | Targets Shape & Dtype | `__getitem__(idx)` Return Format |
| :--- | :--- | :--- | :--- | :--- |
| `SyntheticShapesDataset` | Multi-class shape classification | `(N, C, H, W)` float32 in `[0, 1]` | `(N,)` int64 in `[0, num_classes-1]` | `(image, label)` |
| `SyntheticSegmentationDataset` | Dense multi-class pixel masks | `(N, C, H, W)` float32 in `[0, 1]` | `(N, H, W)` int64 in `[0, num_classes]` | `(image, mask)` |
| `SyntheticDetectionDataset` | Bounding box regression & class IDs | `(N, C, H, W)` float32 in `[0, 1]` | Boxes `(N, M, 4)`, Labels `(N, M)` | `(image, boxes, labels)` |
| `NumerosityDataset` | Discrete object counting | `(N, C, H, W)` float32 in `[0, 1]` | `(N,)` int64 (count offset) | `(image, label)` |
| `SpatialRelationDataset` | Pairwise relational reasoning | `(N, C, H, W)` float32 in `[0, 1]` | `(N,)` int64 in `{0, 1}` | `(image, label)` |
| `TextureVsShapeDataset` | Shape bias vs. texture bias | `(N, 1, H, W)` float32 in `[0, 1]` | Shape `(N,)`, Texture `(N,)` | `(image, shape_label)` |
| `SpatialInvarianceDataset` | Shift invariance & equivariance | `(N, 1, H, W)` float32 | Shifted `(N, 1, H, W)`, Labels `(N,)` | `(base_img, shifted_img, label)` |

#### Detailed Dataset Breakdown & Generation Mechanics

##### `SyntheticShapesDataset`
- **Evaluates**: Multi-class shape classification, spatial pattern recognition, and translation invariance across 8 distinct geometric primitives without external image dependencies.
- **What it generates**:
  1. *Coordinate Meshgrid*: Generates a 2D meshgrid spanning $`[-1.0, 1.0] \times [-1.0, 1.0]`$ over an `img_size x img_size` raster canvas.
  2. *Analytic Shape Geometry*: Uniformly samples a class index from `{0, ..., num_classes - 1}`:
     - `0: Circle`: Euclidean distance $`\sqrt{dx^2 + dy^2} \le r`$
     - `1: Square`: Chebyshev distance $`\max(|dx|, |dy|) \le r`$
     - `2: Triangle`: Upward isosceles triangle of height $`2r`$
     - `3: Cross`: Orthogonal horizontal and vertical bars with arm width $`0.35r`$
     - `4: Rectangle`: Axis-aligned rectangle with aspect ratio 1.8 or 0.55
     - `5: Ellipse`: Smooth quadratic curve $`(dx/rx)^2 + (dy/ry)^2 \le 1.0`$
     - `6: Annulus`: Non-simply connected ring with hole ($`0.45r \le \text{dist} \le r`$)
     - `7: Diamond`: Manhattan $`L_1`$ ball $`|dx| + |dy| \le 1.15r`$
  3. *In-Frame Bounding*: Samples center $`(cx, cy)`$ uniformly in $`[-0.3, 0.3]`$ and radius in $`[0.2, 0.4]`$, guaranteeing the shape is fully contained within the canvas.
  4. *Noise & Channels*: Adds Gaussian background noise $`\mathcal{N}(0, \sigma^2)`$, clamps pixel intensities to $`[0.0, 1.0]`$, and expands across `channels` (1 for grayscale, 3 for RGB).
- **Tensor Layout**:
  - `images`: `(N, channels, img_size, img_size)` float32 in `[0.0, 1.0]`
  - `labels`: `(N,)` int64 in `[0, num_classes - 1]`
- **Concrete Example**:
  With `img_size=32, channels=1, num_classes=8`:
  Generated sample: Class 0 (Circle) with center `(0.10, -0.05)`, radius `0.25`.
  - `images[0]`: shape `(1, 32, 32)` showing a circular white mask on noisy background
  - `labels[0]`: `0`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Images to generate (default: `1000`).
  - `img_size` (`int`): Pixel height and width (default: `32`).
  - `channels` (`int`): Color channels (default: `1`).
  - `noise_std` (`float`): Background noise level (default: `0.1`).
  - `num_classes` (`int`): Shape classes in `[1, 8]` (default: `8`).
  - `seed` (`int`): Random seed (default: `42`).

##### `SyntheticSegmentationDataset`
- **Evaluates**: Dense semantic segmentation architectures (U-Net, FPN, DeepLab, SegFormer), probing spatial detail preservation, encoder-decoder skip connection pathways, and boundary delineation.
- **What it generates**:
  1. *Canvas Initialization*: Allocates an image canvas and an integer class mask of zeros (class `0` is background).
  2. *Sequential Object Rasterization*: Sequentially places `num_objects` randomly sampled shapes onto the canvas.
  3. *Pixel-Wise Ground Truth Mask*: Every pixel covered by a shape is tagged with integer class ID `shape_id + 1` (values in `[1, num_classes]`). Background pixels remain `0`.
  4. *Multi-Channel Formatting*: Background noise is injected, clamped to `[0.0, 1.0]`, and expanded to `channels`.
- **Tensor Layout**:
  - `images`: `(N, channels, img_size, img_size)` float32 in `[0.0, 1.0]`
  - `masks`: `(N, img_size, img_size)` int64 with pixel classes in `[0, num_classes]`
- **Concrete Example**:
  With `img_size=32, num_objects=2, num_classes=3` (1: circle, 2: square, 3: triangle):
  Canvas contains a circle and a triangle.
  - `masks[0][y, x] = 0` for background pixels
  - `masks[0][y, x] = 1` for pixels inside the circle
  - `masks[0][y, x] = 3` for pixels inside the triangle
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Images to generate (default: `1000`).
  - `img_size` (`int`): Pixel height and width (default: `32`).
  - `channels` (`int`): Color channels (default: `1`).
  - `num_classes` (`int`): Foreground classes in `[1, 8]` (default: `8`). Total classes in mask is `num_classes + 1`.
  - `num_objects` (`int`): Shapes per canvas (default: `3`).
  - `noise_std` (`float`): Background noise level (default: `0.1`).
  - `seed` (`int`): Random seed (default: `42`).

##### `SyntheticDetectionDataset`
- **Evaluates**: Object detectors (Faster R-CNN, SSD, YOLO, DETR), testing coordinate regression, anchor box assignment, scale invariance, and bipartite matching.
- **What it generates**:
  1. *Multi-Object Scene Layout*: Renders `num_objects` distinct shapes onto an `(img_size, img_size)` canvas.
  2. *Normalized Bounding Box Derivation*: For each rendered shape, computes the exact bounding box $`[cx, cy, w, h]`$ from its analytic parameters and normalizes coordinates from $`[-1.0, 1.0]`$ to $`[0.0, 1.0]`$.
  3. *Ground Truth Tensors*: Generates paired tensors of normalized bounding box coordinates and corresponding class IDs.
- **Tensor Layout**:
  - `images`: `(N, channels, img_size, img_size)` float32 in `[0.0, 1.0]`
  - `boxes`: `(N, num_objects, 4)` float32 in normalized `[cx, cy, w, h]` format in `[0.0, 1.0]`
  - `labels`: `(N, num_objects)` int64 with shape class IDs in `[0, num_classes - 1]`
- **Concrete Example**:
  With `img_size=32, num_objects=2, num_classes=8`:
  Object 0: Class 0 (Circle), normalized box `[0.35, 0.40, 0.20, 0.20]`.
  Object 1: Class 1 (Square), normalized box `[0.70, 0.65, 0.25, 0.25]`.
  - `boxes[0]`: `[[0.35, 0.40, 0.20, 0.20], [0.70, 0.65, 0.25, 0.25]]`
  - `labels[0]`: `[0, 1]`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Images to generate (default: `1000`).
  - `img_size` (`int`): Pixel height and width (default: `32`).
  - `channels` (`int`): Color channels (default: `1`).
  - `num_classes` (`int`): Shape classes in `[1, 8]` (default: `8`).
  - `num_objects` (`int`): Bounding boxes per image (default: `3`).
  - `noise_std` (`float`): Background noise level (default: `0.1`).
  - `seed` (`int`): Random seed (default: `42`).

##### `NumerosityDataset`
- **Evaluates**: Discrete entity counting without relying on total pixel surface area, testing whether spatial pooling (Global Average Pooling vs. Class Token / Multi-Head Attention) aggregates discrete objects.
- **What it generates**:
  1. *Count Selection*: Samples integer count $`k`$ uniformly in `[min_count, max_count]`.
  2. *Collision-Free Placement*: Positions $`k`$ circular dots with distance constraint $`\text{dist}(c_i, c_j) \ge 2.4 \times \text{radius}`$ to prevent overlap.
  3. *Target Output*: Zero-indexed count class offset `k - min_count`.
- **Tensor Layout**:
  - `images`: `(N, channels, img_size, img_size)` float32 in `[0.0, 1.0]`
  - `labels`: `(N,)` int64 with count offset in `[0, max_count - min_count]`
- **Concrete Example**:
  With `min_count=1, max_count=6, img_size=32`:
  Image generated with 4 scattered non-overlapping dots.
  - `labels[0]`: `4 - 1 = 3` (class 3 corresponds to count 4)
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Images to generate (default: `1000`).
  - `img_size` (`int`): Pixel height and width (default: `32`).
  - `channels` (`int`): Color channels (default: `1`).
  - `min_count` (`int`): Minimum dot count (default: `1`).
  - `max_count` (`int`): Maximum dot count (default: `6`).
  - `radius` (`float`): Radius of each dot in normalized canvas coordinates (default: `0.08`).
  - `noise_std` (`float`): Background noise level (default: `0.1`).
  - `seed` (`int`): Random seed (default: `42`).

##### `SpatialRelationDataset`
- **Evaluates**: Relational spatial reasoning across non-local image patches, diagnosing whether cross-attention or coordinate convolutions can bind spatial locations.
- **What it generates**:
  1. *Two-Entity Scene*: Places Circle at $`(cx_1, cy_1)`$ and Square at $`(cx_2, cy_2)`$ with minimum separation.
  2. *Spatial Predicate Evaluation*:
     - If `relation == "left"`: target is `1` if $`cx_1 \lt cx_2`$ (circle is to the left of square), else `0`.
     - If `relation == "above"`: target is `1` if $`cy_1 \lt cy_2`$ (circle is above square), else `0`.
- **Tensor Layout**:
  - `images`: `(N, channels, img_size, img_size)` float32 in `[0.0, 1.0]`
  - `labels`: `(N,)` int64 in `{0, 1}`
- **Concrete Example**:
  With `relation="left", img_size=32`:
  Circle placed at $`cx_1 = -0.40, cy_1 = 0.00`$; Square placed at $`cx_2 = 0.30, cy_2 = 0.20`$.
  Since $`cx_1 \lt cx_2`$, Circle is to the left of Square.
  - `labels[0]`: `1`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Images to generate (default: `1000`).
  - `img_size` (`int`): Pixel height and width (default: `32`).
  - `channels` (`int`): Color channels (default: `1`).
  - `relation` (`str`): Spatial predicate to test: `"left"` or `"above"` (default: `"left"`).
  - `noise_std` (`float`): Background noise level (default: `0.1`).
  - `seed` (`int`): Random seed (default: `42`).

##### `TextureVsShapeDataset`
- **Evaluates**: Shape bias vs. texture bias (Geirhos et al.), quantifying whether an architecture relies on boundary geometry (human/ViT bias) or high-frequency surface textures (standard CNN bias).
- **What it generates**:
  1. *Shape Mask*: Primary silhouette (0: Circle of radius 0.6, 1: Square of half-width 0.6).
  2. *Orthogonal Texture*: Procedural stripe pattern (0: Horizontal square-wave, 1: Vertical square-wave).
  3. *Modulation*: Inside shape boundary, pixel intensity is `0.3 + 0.7 * texture_pattern`. Background is `0.0`.
  4. *Dual Target Labels*: Exposes `.shape_labels` and `.texture_labels`.
- **Tensor Layout**:
  - `images`: `(N, 1, img_size, img_size)` float32 in `[0.0, 1.0]`
  - `shape_labels`: `(N,)` int64 in `{0, 1}` (0: Circle, 1: Square)
  - `texture_labels`: `(N,)` int64 in `{0, 1}` (0: Horizontal, 1: Vertical)
- **Concrete Example**:
  With `img_size=32`:
  Sample generated: Shape 0 (Circle), Texture 1 (Vertical stripes).
  - `images[0]`: Circular mask filled with alternating black and white vertical zebra stripes
  - `shape_labels[0]`: `0`
  - `texture_labels[0]`: `1`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Samples to generate (default: `1000`).
  - `img_size` (`int`): Pixel height and width (default: `32`).
  - `seed` (`int`): Random seed (default: `42`).

##### `SpatialInvarianceDataset`
- **Evaluates**: Translation invariance and equivariance under discrete spatial coordinate shifts.
- **What it generates**:
  1. *Base Image*: Centered shape (Circle or Square) at $`(cx=0, cy=0)`$.
  2. *Shifted Image*: Identical shape translated horizontally along the X-axis by `shift_pixels`.
  3. *Returns*: Paired tensors `(base_image, shifted_image, label)` to quantify feature distortion $`\frac{\|f(T(X)) - f(X)\|_F}{\|f(X)\|_F}`$.
- **Tensor Layout**:
  - `base_images`: `(N, 1, img_size, img_size)` float32
  - `shifted_images`: `(N, 1, img_size, img_size)` float32 translated by `shift_pixels`
  - `labels`: `(N,)` int64 in `{0, 1}`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Image pairs to generate (default: `1000`).
  - `img_size` (`int`): Pixel height and width (default: `32`).
  - `shift_pixels` (`int`): Pixel translation distance (default: `4`).
  - `seed` (`int`): Random seed (default: `42`).

```python
from torch.utils.data import DataLoader
from syndset.domains.vision import (
  SyntheticDetectionDataset,
  SyntheticSegmentationDataset,
  SyntheticShapesDataset,
)

# 1. Classification
shapes_data = SyntheticShapesDataset(num_samples=500, img_size=32, channels=1, num_classes=8)
img, label = shapes_data[0]
print(f"Shape image: {img.shape}, class: {label}")

# 2. Semantic Segmentation
seg_data = SyntheticSegmentationDataset(num_samples=500, img_size=32, num_classes=8, num_objects=3)
img, mask = seg_data[0]
print(f"Seg image: {img.shape}, mask: {mask.shape}, unique classes in mask: {mask.unique().tolist()}")

# 3. Object Detection
det_data = SyntheticDetectionDataset(num_samples=500, img_size=32, num_objects=3)
img, boxes, labels = det_data[0]
print(f"Det boxes: {boxes.shape} (normalized [cx, cy, w, h]), labels: {labels.shape}")
```

---

### Tabular & Dense Models (`syndset.domains.tabular`)

Evaluates non-linear decision boundary untangling, coordinate sparsity, ill-conditioned curvature, and topological separation.

| Dataset | Evaluates | `data` Shape & Dtype | Targets Shape & Dtype | `__getitem__(idx)` Return Format |
| :--- | :--- | :--- | :--- | :--- |
| `ConcentricHyperspheresDataset` | Non-linear depth efficiency | `(N, dim)` float32 | `(N,)` int64 in `[0, num_classes-1]` | `(features, label)` |
| `SparseXORDataset` | Feature selection & distractor noise | `(N, total_dim)` float32 | `(N,)` int64 in `{0, 1}` | `(features, parity_label)` |
| `IllConditionedRegressionDataset` | Condition number $`\kappa \ge 10^4`$ | `(N, dim)` float32 | `(N,)` float32 | `(features, target_scalar)` |
| `TwoSpiralsDataset` | Topological manifold untangling | `(N, 2)` float32 | `(N,)` int64 in `{0, 1}` | `(coord_2d, spiral_id)` |

#### Detailed Dataset Breakdown & Generation Mechanics

##### `ConcentricHyperspheresDataset`
- **Evaluates**: Non-linear representational capacity and depth efficiency in feedforward networks / MLPs on linearly inseparable concentric topological manifolds.
- **What it generates**:
  1. *Spherical Shell Manifolds*: Partitions `num_samples` points uniformly across `num_classes` concentric hyperspherical shells in $`\mathbb{R}^{\text{dim}}`$.
  2. *Isotropic Direction Sampling*: Normalizes Gaussian vectors to obtain uniform random points on the unit hypersphere: $`u = z / (\|z\|_2 + 10^{-12})`$, where $`z \sim \mathcal{N}(0, I_{\text{dim}})`$.
  3. *Radial Shell Scaling*: Assigns base radius for shell $`c`$ as $`r_c = c + 1`$ (e.g. 1.0, 2.0, 3.0), adds radial Gaussian noise with standard deviation `noise_std` (clamped to $`\ge 0.1`$), and multiplies unit directions: $`x = u \cdot r`$.
  4. *Target Output*: Shell class index $`c`$ in `[0, num_classes - 1]`.
- **Tensor Layout**:
  - `data`: `(N, dim)` float32
  - `labels`: `(N,)` int64 with class indices in `[0, num_classes - 1]`
- **Concrete Example**:
  With `dim=4, num_classes=3, noise_std=0.02`:
  - Shell 0 point (radius ~1.0): `[ 0.52, -0.61,  0.33, -0.49 ]` -> `label: 0`
  - Shell 1 point (radius ~2.0): `[-1.15,  0.88,  1.20, -0.42 ]` -> `label: 1`
  - Shell 2 point (radius ~3.0): `[ 1.80,  1.55, -1.22,  1.05 ]` -> `label: 2`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Total points (default: `1000`).
  - `num_classes` (`int`): Concentric spherical shells (default: `3`). Radii: $`1.0, 2.0, \dots`$.
  - `dim` (`int`): Feature space dimension (default: `8`).
  - `noise_std` (`float`): Radial perturbation noise (default: `0.05`).
  - `seed` (`int`): Random seed (default: `42`).

##### `SparseXORDataset`
- **Evaluates**: Coordinate selection and high-order feature interactions, testing whether an architecture can extract informative sparse signal coordinates in the presence of dominant noise distractors.
- **What it generates**:
  1. *Informative Signal Coordinates*: Samples `active_dim` binary bits in $`\{0, 1\}`$, maps them to $`\{-1.0, +1.0\}`$, and assigns them to the first `active_dim` coordinates.
  2. *Distractor Noise Floor*: Fills the remaining `total_dim - active_dim` coordinates with uninformative standard Gaussian noise $`\mathcal{N}(0, 1)`$.
  3. *Parity Target*: Target is the XOR parity of the active bits: $`(\sum \text{bits}) \bmod 2 \in \{0, 1\}`$.
- **Tensor Layout**:
  - `data`: `(N, total_dim)` float32 (first `active_dim` in $`\{-1.0, +1.0\}`$, remaining in $`\mathcal{N}(0, 1)`$)
  - `labels`: `(N,)` int64 in `{0, 1}`
- **Concrete Example**:
  With `total_dim=6, active_dim=3`:
  Active bits `[1, 0, 1]` -> signal coordinates `[+1.0, -1.0, +1.0]`. Distractor noise `[-0.43, 1.25, 0.08]`.
  - `data[0]`: `[ +1.0, -1.0, +1.0, -0.43, 1.25, 0.08 ]`
  - `labels[0]`: `(1 + 0 + 1) % 2 = 0`
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Total points (default: `1000`).
  - `total_dim` (`int`): Total features, including Gaussian distractors (default: `64`).
  - `active_dim` (`int`): Informative binary coordinates determining parity (default: `3`).
  - `seed` (`int`): Random seed (default: `42`).

##### `IllConditionedRegressionDataset`
- **Evaluates**: Optimizer convergence, gradient curvature, and loss conditioning under extreme feature multi-collinearity ($`\kappa \ge 10^4`$).
- **What it generates**:
  1. *Geometrically Decaying Spectrum*: Constructs singular values $`\sigma_i`$ decaying exponentially from $`1.0`$ down to $`1/\kappa`$: $`\sigma_i = 10^{-i \log_{10}(\kappa) / (\text{dim} - 1)}`$.
  2. *Random Orthogonal Basis*: Projects singular spectrum through an orthonormal matrix $`Q`$ from QR decomposition: $`\text{cov}^{1/2} = Q \, \text{diag}(\sigma)`$.
  3. *Correlated Inputs*: Correlates white noise $`Z \sim \mathcal{N}(0, I)`$ via $`X = Z \cdot (\text{cov}^{1/2})^\top`$.
  4. *Linear Targets*: Generates continuous targets $`y = X w + \epsilon`$, where $`w \sim \mathcal{N}(0, I)`$ and $`\epsilon \sim \mathcal{N}(0, \sigma_{\text{noise}}^2)`$.
- **Tensor Layout**:
  - `data`: `(N, dim)` float32 with ill-conditioned covariance matrix
  - `targets`: `(N,)` float32 continuous regression targets
- **Concrete Example**:
  With `dim=4, condition_number=1e4`:
  Singular values: `[1.0, 0.0464, 0.00215, 0.0001]`. Condition number $`\kappa = 10,000`$.
  Gradient updates along the smallest eigenvalue axis will be $`10,000\times`$ slower than along the principal axis unless adaptive preconditioning is used.
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Samples to generate (default: `1000`).
  - `dim` (`int`): Feature dimension (default: `32`).
  - `condition_number` (`float`): Covariance matrix singular value condition ratio $`\kappa = \sigma_{\max} / \sigma_{\min}`$ (default: `1e4`).
  - `noise_std` (`float`): Additive target noise (default: `0.01`).
  - `seed` (`int`): Random seed (default: `42`).

##### `TwoSpiralsDataset`
- **Evaluates**: Non-linear classification, decision boundary winding, and topological manifold untangling.
- **What it generates**:
  1. *Archimedean Trajectory*: Continuous spiral radius $`r(\theta) = \theta / (2\pi \cdot \text{turns})`$ for $`\theta \in [0, 2\pi \cdot \text{turns}]`$.
  2. *Interlocked Dual Arms*:
     - Spiral 1 (class 0): $`x_1 = r(\theta) \cos(\theta) + \epsilon, y_1 = r(\theta) \sin(\theta) + \epsilon`$.
     - Spiral 2 (class 1, rotated by $`\pi`$): $`x_2 = -r(\theta) \cos(\theta) + \epsilon, y_2 = -r(\theta) \sin(\theta) + \epsilon`$.
  3. *Topological Inseparability*: The two branches wind around each other `turns` times without intersecting.
- **Tensor Layout**:
  - `data`: `(N, 2)` float32 2D coordinates
  - `labels`: `(N,)` int64 in `{0, 1}`
- **Concrete Example**:
  With `turns=2.5, noise_std=0.05`:
  - `data[0]`: `[ 0.28, -0.74 ]` -> `labels[0]: 0` (Spiral branch 1)
  - `data[1]`: `[-0.31,  0.71 ]` -> `labels[1]: 1` (Spiral branch 2)
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): 2D points to generate (default: `1000`).
  - `turns` (`float`): Revolutions per Archimedean spiral arm (default: `2.5`).
  - `noise_std` (`float`): Cartesian coordinate noise (default: `0.05`).
  - `seed` (`int`): Random seed (default: `42`).

```python
from syndset.domains.tabular import (
  ConcentricHyperspheresDataset,
  IllConditionedRegressionDataset,
  SparseXORDataset,
  TwoSpiralsDataset,
)

# 1. Concentric Hyperspheres
spheres = ConcentricHyperspheresDataset(num_samples=1000, num_classes=3, dim=16)
x, y = spheres[0]
print(f"Features: {x.shape}, Shell class: {y.item()}")

# 2. Ill-Conditioned Regression (kappa = 10^4)
ill_data = IllConditionedRegressionDataset(num_samples=1000, dim=32, condition_number=1e4)
feat, target = ill_data[0]
print(f"Ill-conditioned feature: {feat.shape}, target: {target.item():.4f}")

# 3. Two Spirals
spirals = TwoSpiralsDataset(num_samples=1000, turns=2.5)
coord, label = spirals[0]
print(f"Spiral 2D coord: {coord.tolist()}, branch: {label.item()}")
```

---

### Time Series Models (`syndset.domains.timeseries`)

Evaluates spectral frequency resolution, high-frequency dampening (spectral bias), and distant temporal lag dependencies.

| Dataset | Evaluates | `inputs` / `sequences` Shape & Dtype | `targets` Shape & Dtype | `__getitem__(idx)` Return Format |
| :--- | :--- | :--- | :--- | :--- |
| `HarmonicSuperpositionDataset` | Spectral bias across Fourier harmonics | `(N, seq_len - 1, 1)` float32 | `(N, seq_len - 1, 1)` float32 | `(input_seq, target_seq)` |
| `AutoregressiveLagDataset` | Long-range historical lag preservation | `(N, total_steps - 1, 1)` float32 | `(N, total_steps - 1, 1)` float32 | `(input_seq, target_seq)` |

#### Detailed Dataset Breakdown & Generation Mechanics

##### `HarmonicSuperpositionDataset`
- **Evaluates**: Spectral bias (F-Principle), verifying whether temporal, convolutional, or spectral architectures can capture high-frequency components or succumb to low-pass dampening.
- **What it generates**:
  1. *Temporal Time Mesh*: Evaluates continuous time steps across `seq_len` points: $`t \in [0.0, 1.0]`$.
  2. *Multi-Harmonic Superposition*: Superimposes `num_harmonics` sinusoidal waves with harmonic frequencies and power-law amplitude decay:
     For harmonic $`h \in \{1, \dots, H\}`$ (where $`H =`$ `num_harmonics`):
     - Frequency $`f_h = 2h`$
     - Amplitude $`a_h = 1 / \sqrt{h}`$
     - Random phase $`\phi_h \sim \text{Uniform}(0, 2\pi)`$
     - Continuous signal: $`\text{signal}(t) = \sum_{h=1}^H a_h \sin(2\pi f_h t + \phi_h)`$
  3. *Noise Injection & 1-Step Forecast Split*:
     - Injects additive Gaussian noise: `noisy_signal = signal + noise` where noise has standard deviation `noise_std`.
     - Inputs: past window `noisy_signal[:-1]` (shape: `(seq_len - 1, 1)`).
     - Targets: clean 1-step shifted forecast `signal[1:]` (shape: `(seq_len - 1, 1)`).
- **Tensor Layout**:
  - `inputs`: `(N, seq_len - 1, 1)` float32 with noisy past values
  - `targets`: `(N, seq_len - 1, 1)` float32 with clean 1-step future values
- **Concrete Example**:
  With `seq_len=8, num_harmonics=2`:
  - `inputs[0]`: 7 time steps of noisy signal $`[x_0, x_1, x_2, x_3, x_4, x_5, x_6]`$
  - `targets[0]`: 7 time steps of clean future $`[s_1, s_2, s_3, s_4, s_5, s_6, s_7]`$
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `seq_len` (`int`): Time steps per sequence (default: `64`). Model input sequence length is `seq_len - 1`.
  - `num_harmonics` (`int`): Sinusoidal Fourier components superimposed (default: `4`).
  - `noise_std` (`float`): Additive Gaussian noise (default: `0.05`).
  - `seed` (`int`): Random seed (default: `42`).

##### `AutoregressiveLagDataset`
- **Evaluates**: Long-range temporal lag dependencies, testing whether recurrent, TCN, or state-space models can retain memory over distant horizons without exponential gradient decay.
- **What it generates**:
  1. *Prefix Initialization*: Initializes a sequence of length `total_steps`, seeding the initial prefix `0 : max_lag` with standard normal values $`\mathcal{N}(0, 0.25)`$.
  2. *Non-Linear Delayed Lag Dynamics*:
     For time step $`t`$ from `max_lag` to `total_steps - 1`:
     - Computes linear combination of specified historical lag values:
       $`\text{val} = \sum_l w_l \cdot \text{series}[t - \text{lag}_l]`$ (alternating weights $`+0.5, -0.4`$).
     - Applies non-linear saturation and noise (with standard deviation `noise_std`):
       $`\text{series}[t] = \tanh(\text{val}) + \epsilon`$.
  3. *Shifted Forecast Targets*:
     - `sequences`: past sequence `series[:-1]` of length `total_steps - 1`.
     - `targets`: 1-step shifted sequence `series[1:]` of length `total_steps - 1`.
- **Tensor Layout**:
  - `sequences`: `(N, total_steps - 1, 1)` float32
  - `targets`: `(N, total_steps - 1, 1)` float32
- **Concrete Example**:
  With `total_steps=32, lags=(5, 20)`:
  Step $`t=20`$ explicitly depends on step $`t=15`$ (lag 5) and step $`t=0`$ (lag 20).
  Tests if the architecture can bridge a 20-step temporal gap to accurately forecast future steps.
- **Constructor Parameters (`__init__`)**:
  - `num_samples` (`int`): Sequences to generate (default: `1000`).
  - `total_steps` (`int`): Total steps (default: `64`). Input sequence length is `total_steps - 1`.
  - `lags` (`tuple[int, ...]`): Explicit governing historical time lag indices (default: `(5, 20)`).
  - `noise_std` (`float`): Additive noise level (default: `0.02`).
  - `seed` (`int`): Random seed (default: `42`).

```python
from torch.utils.data import DataLoader
from syndset.domains.timeseries import AutoregressiveLagDataset, HarmonicSuperpositionDataset

# 1. Harmonic Superposition
waves = HarmonicSuperpositionDataset(num_samples=500, seq_len=64, num_harmonics=4)
x, y = waves[0]
print(f"Harmonic wave input: {x.shape}, forecast target: {y.shape}")

# 2. Autoregressive Lags
lags_data = AutoregressiveLagDataset(num_samples=500, total_steps=64, lags=(5, 20))
x, y = lags_data[0]
print(f"Lag input: {x.shape}, target: {y.shape}")
```

---


---

## Mathematical Foundations & Theoretical Mechanics

The analytical diagnostics and synthetic task generators in `syndset` are built on exact mathematical formulations, analytical ground-truth bounds, and information-theoretic principles.

To ensure instant browser rendering and avoid GitHub per-page math complexity limits, complete mathematical derivations, condition bounds, and proofs are modularized into dedicated technical references:

### Technical Reference Documents:
* 🔬 **[Diagnostic Probes & Decision Mechanics](docs/theory_diagnostics.md)**: Derivations, singular value entropy bounds, gradient flow condition numbers, CKA Hilbert-Schmidt independence criteria, SynFlow conservation, NTK eigenspectra, and TOPSIS Pareto dominance mechanics.
* 📊 **[Synthetic Benchmark Tasks & Inductive Biases](docs/theory_datasets.md)**: Analytical sequence/spatial data-generating mechanics, manifold geometries (hyperspheres, two-spirals), dyck languages, induction copying circuits, and causal support masks.

---

### Core Theoretical Frameworks

| Domain | Analytical Diagnostic | Core Theoretical Invariant / Metric | Primary Failure Regime | Reference |
| :--- | :--- | :--- | :--- | :--- |
| **Geometry** | Effective Rank | $`\text{erank}(H) = \exp(-\sum p_i \ln p_i)`$ | Dimensional collapse ($`\text{erank} \ll D`$) | [Derivation](docs/theory_diagnostics.md#1-effective-rank--dimensional-collapse) |
| **Gradients** | Flow & Conditioning | $`R_{\text{vanish}} = \|\nabla_{\theta_1} \mathcal{L}\| / \|\nabla_{\theta_L} \mathcal{L}\|`$ | Vanishing / exploding gradients | [Derivation](docs/theory_diagnostics.md#2-gradient-flow--jacobian-conditioning) |
| **Topology** | Dead Units | $`\rho_{\text{dead}} = \frac{1}{M} \sum \mathbf{1}_{\{\max |a_i| \le \epsilon\}}`$ | Dead ReLU saturation ($`\gt 10\%`$) | [Derivation](docs/theory_diagnostics.md#4-dead-units--activation-saturation) |
| **Precision** | Dynamic Range | $`\delta_{\text{IEEE}} = \|f(x) - f_{\text{half}}(x)\|_F / \|f(x)\|_F`$ | FP16 overflow / underflow | [Derivation](docs/theory_diagnostics.md#5-floating-point-dynamic-range--mixed-precision-perturbations) |
| **Curvature** | Loss Sharpness | $`\text{Tr}(H) \approx \frac{1}{M} \sum v_i^\top \nabla_\theta (\nabla_\theta \mathcal{L}^\top v_i)`$ | Ill-conditioned curvature ($`\gt 10^4`$) | [Derivation](docs/theory_diagnostics.md#7-loss-surface-curvature--hutchinsons-hessian-trace-estimator) |
| **Attribution** | Causal SNR | $`\text{SNR}_{\text{attr}} = 10 \log_{10} (\mu_{\text{causal}}^2 / \mu_{\text{noise}}^2)`$ | Inverted feature selection ($`\lt 0 \text{ dB}`$) | [Derivation](docs/theory_diagnostics.md#10-axiomatic-attribution--causal-ground-truth-mechanics) |
| **Similarity** | Linear CKA | $`\text{CKA}(X, Y) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \text{HSIC}(L, L)}}`$ | Depth redundancy ($`\gt 0.99`$) | [Derivation](docs/theory_diagnostics.md#11-centered-kernel-alignment-cka--representation-dynamics) |
| **Pruning** | SynFlow | $`\mathcal{R}(\theta) = \mathbf{1}^\top f(\mathbf{1}; |\theta|)`$ | Topological disconnection / starvation | [Derivation](docs/theory_diagnostics.md#12-data-free-synaptic-flow-conservation-synflow) |
| **Quantization**| Kurtosis & Outliers | $`\text{Kurt}(X) = \frac{\mu_4}{\sigma_X^4}`$ | Heavy tails collapsing INT8/FP8 grids | [Derivation](docs/theory_diagnostics.md#13-activation-kurtosis--low-precision-ptq-viability) |
| **Dynamics** | Empirical NTK | $`\kappa(\Theta) = \lambda_1 / \lambda_N`$ | Optimization stiffness ($`\gt 10^5`$) | [Derivation](docs/theory_diagnostics.md#14-empirical-neural-tangent-kernel-ntk-conditioning) |
| **Receptive** | Normalized ERF | $`\rho_{\text{ERF}} = 2\sqrt{\sigma_u^2 + \sigma_v^2} / \max(H, W)`$ | Constricted field ($`\lt 0.15`$) | [Derivation](docs/theory_diagnostics.md#15-effective-receptive-field-erf-spatial-dispersion) |
| **Selection** | Pareto & TOPSIS | $`U_i = d_i^- / (d_i^+ + d_i^-)`$ | Suboptimal trade-offs (veto / dominated)| [Derivation](docs/theory_diagnostics.md#16-multi-criteria-pareto-dominance--decision-mechanics) |


---


## Development & Publishing

### Running Tests
```bash
pytest -v
```

### Code Formatting & Linting
The codebase follows **Google Python Style** with **2-space indentation**, formatted using `yapf` (configured via `.style.yapf`):

```bash
# Check formatting with YAPF
yapf --diff --recursive syndset tests

# Apply formatting in-place
yapf --in-place --recursive syndset tests

# Run linter checks with Ruff
ruff check .
```

### Building & Publishing to PyPI
```bash
python3 -m pip install build twine
python3 -m build
twine check dist/*
twine upload dist/*
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
