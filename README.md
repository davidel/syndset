# syndset

[![CI](https://github.com/davidel/syndset/actions/workflows/ci.yml/badge.svg)](https://github.com/davidel/syndset/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/syndset/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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

## Diagnostic Probes Overview

Every diagnostic probe attaches non-invasively to any standard PyTorch `nn.Module`. For complete mathematical derivations, condition numbers, and failure regimes, see the [Mathematical Foundations & Theoretical Mechanics](#mathematical-foundations--theoretical-mechanics) section below.

### 1. Gradient Flow (`syn.check_gradient_flow`)
Hooks into backpropagation on a single step to verify that gradients reach every parameter.

```python
results = syn.check_gradient_flow(model, sample_input, target=sample_target)
print("Status:", results["status"])
print("Vanishing ratio:", results["vanishing_ratio"])
print("Unused parameters:", results["zero_grad_params"])
```

* **Detects:** Disconnected parameters ($\|\nabla_\theta \mathcal{L}\| = 0$), exponential gradient vanishing ($R_{\text{vanish}} < 10^{-5}$), and explosive gradient growth.
* [Jump to Gradient Flow Math & Derivation &rarr;](#2-gradient-flow--jacobian-conditioning)

### 2. Effective Rank & Dimensional Collapse (`syn.check_effective_rank`)
Computes the spectral entropy-based effective rank of hidden layers using Singular Value Decomposition:

```python
rank_info = syn.check_effective_rank(model, sample_input)
print("Layer ranks:", rank_info["layer_ranks"])
print("Collapsed layers:", rank_info["collapsed_layers"])
```

* **Detects:** Representation collapse (where high-dimensional representations flatten into a 1D/2D subspace) and attention anisotropy.
* [Jump to Effective Rank Math & SVD Formulation &rarr;](#1-effective-rank--dimensional-collapse)

### 3. Dead Units Probe (`syn.check_dead_units`)
Measures the variance of activations across batch samples.

```python
dead_info = syn.check_dead_units(model, sample_input)
print("Dead ratios per layer:", dead_info["layer_dead_fractions"])
```

* **Detects:** Saturated non-linearities and the classic "dying ReLU" failure mode where neurons receive zero gradient across all samples.
* [Jump to Dead Units Formulation &rarr;](#4-dead-units--activation-saturation)

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

* **Detects:** Activations exceeding IEEE 754 FP16 limits ($65,504$) due to unscaled dot products, and catastrophic cancellation in normalization layers.
* [Jump to Mixed-Precision Math & IEEE 754 Details &rarr;](#5-floating-point-dynamic-range--mixed-precision-perturbations)

---

## Synthetic Benchmark Datasets Overview

`syndset` provides procedural datasets designed to test specific algorithmic and geometric capabilities without external downloads. Detailed theoretical motivations for each task are provided in the [Synthetic Task Theory & Inductive Biases](#6-synthetic-task-theory--inductive-biases) section below.

### LLM & Sequence Models (`syndset.domains.llm`)

| Dataset | What It Evaluates |
| :--- | :--- |
| `AssociativeRecallDataset` | Key-value associative binding and retrieval over sequence horizons (`[k1, v1, ..., kq] -> vq`). [Details &rarr;](#associative-recall--key-value-retrieval) |
| `InductionDataset` | In-context pattern completion (`[A, B, ..., A] -> B`), testing induction head mechanics. [Details &rarr;](#induction-heads-a--b--a--b) |
| `SelectiveCopyDataset` | Distractor noise filtering and ordered memory reproduction. [Details &rarr;](#selective-copying) |
| `CumulativeParityDataset` | Step-by-step cumulative XOR tracking to verify discrete state compositionality. [Details &rarr;](#cumulative-parity) |

```python
from syndset.domains.llm import AssociativeRecallDataset

# 1000 sequences, 8 key-value pairs per sequence, vocab size 64
data = AssociativeRecallDataset(num_samples=1000, num_pairs=8, vocab_size=64)
tokens, target = data[0]
```

### Vision & Spatial Models (`syndset.domains.vision`)

| Dataset | What It Evaluates |
| :--- | :--- |
| `SyntheticShapesDataset` | Multi-class geometric shapes with controlled variations in scale, position, color, and noise. |
| `TextureVsShapeDataset` | Pits low-frequency shape silhouettes against conflicting high-frequency texture stripes to measure inductive bias. |
| `SpatialInvarianceDataset` | Shifted image pairs to quantify translation invariance and coordinate equivariance. |

### Tabular & Dense Models (`syndset.domains.tabular`)

| Dataset | What It Evaluates |
| :--- | :--- |
| `ConcentricHyperspheresDataset` | Nested $D$-dimensional spherical shells to test non-linear decision boundary depth efficiency. [Details &rarr;](#concentric-hyperspheres--manifold-curvature) |
| `SparseXORDataset` | High-dimensional inputs where only $k$ sparse coordinates determine target parity, testing coordinate selection. |
| `IllConditionedRegressionDataset` | Covariance matrix with condition number $\kappa = 10^4+$, testing optimization conditioning. [Details &rarr;](#ill-conditioned-regression--optimization-curvature) |

### Time Series Models (`syndset.domains.timeseries`)

| Dataset | What It Evaluates |
| :--- | :--- |
| `HarmonicSuperpositionDataset` | Multi-frequency Fourier sums testing whether models capture high frequencies (spectral bias). [Details &rarr;](#spectral-bias--harmonic-superposition) |
| `AutoregressiveLagDataset` | Non-linear time series with explicit distant lag dependencies to verify long-term memory. |

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

## Mathematical Foundations & Theoretical Mechanics

This section provides the mathematical derivations, condition numbers, spectral theorems, and empirical references underlying `syndset`.

### 1. Effective Rank & Dimensional Collapse

#### The Problem: Anisotropy and Representation Collapse
In deep architectures—especially Transformers without proper residual scaling or self-supervised representations without contrastive repulsion—hidden representations often suffer from **anisotropy** or **dimensional collapse** (Dong et al., 2021, *"Attention is not all you need"*). In this degenerate state, the network projects all $D$-dimensional hidden states into a tiny subspace (e.g., a 1D line or narrow cone in $\mathbb{R}^D$), rendering the vast majority of parameters useless.

#### The Mathematical Formulation
Let $H \in \mathbb{R}^{N \times D}$ denote the matrix of activations from a hidden layer across $N$ samples (or $N = B \times T$ token representations in sequence models), with hidden dimension $D$.

We compute the Singular Value Decomposition (SVD):
$$H = U \Sigma V^\top$$
where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \dots, \sigma_r)$ with non-zero singular values ordered as $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$, where $r = \min(N, D)$.

Standard linear algebraic rank counts non-zero singular values:
$$\text{rank}(H) = \sum_{i=1}^r \mathbf{1}_{\{\sigma_i > 0\}}$$
However, standard rank is an unreliable discontinuous step function in numerical computing because floating-point precision noise introduces tiny non-zero singular values ($\sim 10^{-16}$).

To capture genuine continuous dimensionality, `syndset` implements **Spectral Entropy Effective Rank** (Roy & Vetterli, 2007):

1. **Normalize the singular values** into a probability distribution over the singular modes:
   $$p_i = \frac{\sigma_i}{\sum_{j=1}^r \sigma_j}, \quad \text{such that} \quad \sum_{i=1}^r p_i = 1$$

2. **Compute the Shannon spectral entropy:**
   $$H(p) = -\sum_{i=1}^r p_i \ln p_i$$

3. **Compute the effective rank:**
   $$\text{erank}(H) = \exp\left(H(p)\right)$$

#### Theoretical Bounds & Interpretation
* **Maximum Dimensionality (Isotropic Dispersion):**
  If all singular values are equal ($\sigma_1 = \sigma_2 = \dots = \sigma_r$), then $p_i = 1/r$ for all $i$:
  $$H(p) = -\sum_{i=1}^r \frac{1}{r} \ln\left(\frac{1}{r}\right) = \ln r \implies \text{erank}(H) = \exp(\ln r) = r = \min(N, D)$$
  The layer utilizes 100% of its available representation space.

* **Total Dimensional Collapse:**
  If a single singular value dominates all others ($\sigma_1 \gg \sigma_2 \approx 0$), then $p_1 \to 1$ and $p_{i>1} \to 0$:
  $$H(p) \to 0 \implies \text{erank}(H) \to \exp(0) = 1$$
  Even if $D = 4096$, the entire batch is confined to a 1-dimensional line.

`syndset` monitors the **rank utilization ratio**:
$$\rho_{\text{rank}} = \frac{\text{erank}(H)}{\min(N, D)}$$
Layers with $\rho_{\text{rank}} < 0.05$ (less than 5% capacity utilization) trigger an architectural warning for dimensional collapse.

---

### 2. Gradient Flow & Jacobian Conditioning

#### The Multivariable Chain Rule
Consider an $L$-layer neural network parameterized by $\{\theta_1, \dots, \theta_L\}$:
$$h_l = f_l(h_{l-1}; \theta_l), \quad l = 1, \dots, L$$
where $h_0 = x$ is the network input and $h_L = \hat{y}$ is the output. Let $\mathcal{L}(h_L, y)$ be a scalar loss function.

By the chain rule of multivariable calculus, the backpropagated gradient with respect to the input of layer $l$ is:
$$\frac{\partial \mathcal{L}}{\partial h_{l-1}} = \frac{\partial \mathcal{L}}{\partial h_l} J_l$$
where $J_l \in \mathbb{R}^{\dim(h_l) \times \dim(h_{l-1})}$ is the layer Jacobian matrix:
$$[J_l]_{jk} = \frac{\partial [h_l]_j}{\partial [h_{l-1}]_k}$$

Iterating from the loss back to the input layer $h_0$:
$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^L J_l$$

Applying the submultiplicative property of induced matrix 2-norms (spectral norm, equivalent to the maximum singular value $\sigma_{\max}$):
$$\left\|\frac{\partial \mathcal{L}}{\partial h_0}\right\|_2 \le \left\|\frac{\partial \mathcal{L}}{\partial h_L}\right\|_2 \prod_{l=1}^L \|J_l\|_2$$

#### The Failure Regimes
1. **Exponential Gradient Vanishing:**
   If the spectral norm of the layer Jacobians is bounded strictly below unity, $\|J_l\|_2 \le 1 - \epsilon$:
   $$\left\|\frac{\partial \mathcal{L}}{\partial h_0}\right\|_2 \le \mathcal{O}\left((1 - \epsilon)^L\right) \xrightarrow{L \to \infty} 0$$
   The bottom layers receive virtually zero gradient, leaving early representations at their initial random state. `syndset` computes the **vanishing ratio**:
   $$R_{\text{vanish}} = \frac{\min_l \|\nabla_{\theta_l} \mathcal{L}\|_2}{\max_l \|\nabla_{\theta_l} \mathcal{L}\|_2 + 10^{-12}}$$
   If $R_{\text{vanish}} < 10^{-5}$, an architectural gradient vanishing failure is flagged.

2. **Exponential Gradient Explosion:**
   If $\|J_l\|_2 \ge 1 + \epsilon$, the gradient norm grows as $\mathcal{O}((1+\epsilon)^L) \to \infty$, causing optimizer parameter divergence and numerical overflow (`inf`/`nan`).

3. **Disconnected Subgraphs & Dead Parameters:**
   If a trainable parameter tensor $\theta_k$ receives an exact gradient norm of $\|\nabla_{\theta_k} \mathcal{L}\|_2 = 0$ while subsequent layers receive non-zero gradients, it identifies a structural graph disconnect (e.g., an unintended `.detach()`, a misplaced return statement, or an unused conditional branch).

---

### 3. Gradient Signal-to-Noise Ratio (SNR)

In minibatch gradient descent, the parameter gradient evaluated on minibatch $B_k$ is a stochastic estimator $g_k = \nabla_\theta \mathcal{L}(B_k)$ of the true population gradient $\mathbb{E}[\nabla_\theta \mathcal{L}]$.

Over $K$ minibatches, `syndset` calculates:
$$\bar{g} = \frac{1}{K} \sum_{k=1}^K g_k, \quad s^2 = \frac{1}{K-1} \sum_{k=1}^K (g_k - \bar{g})^{\odot 2}$$
The aggregate parameter SNR is:
$$\text{SNR}(\theta) = \frac{\|\bar{g}\|_2}{\sqrt{\sum_i s_i^2} + 10^{-10}}$$

* **$\text{SNR} > 1.0$ (Signal-Dominated):** Minibatches agree on a consistent descent direction.
* **$\text{SNR} < 0.1$ (Noise-Dominated):** Gradient updates are dominated by stochastic variance rather than systematic descent, indicating that the chosen batch size is inadequate or the loss surface is excessively noisy.

---

### 4. Dead Units & Activation Saturation

For a feature channel or neuron $j$, let $h_{b, j}$ denote its activation for sample $b$ in a batch of size $B$. The sample variance across the batch is:
$$\text{Var}_B(h_j) = \frac{1}{B - 1} \sum_{b=1}^B (h_{b, j} - \bar{h}_j)^2, \quad \bar{h}_j = \frac{1}{B} \sum_{b=1}^B h_{b, j}$$

A unit is classified as **dead** if $\text{Var}_B(h_j) < 10^{-8}$.

#### The Dying ReLU Phenomenon
For activation $f(z) = \max(0, z)$, the subgradient is:
$$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$
If a neuron's weights are updated such that its pre-activation $z_{b, j} \le 0$ for all $b \in B$, then:
$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \sum_{b=1}^B \frac{\partial \mathcal{L}}{\partial h_{b, j}} \cdot f'(z_{b, j}) \cdot x_{b, i} = 0$$
The neuron produces a constant output of 0 and receives a gradient of 0, making it permanently incapable of recovering. `syndset` flags any layer where more than 30% of units are dead.

---

### 5. Floating-Point Dynamic Range & Mixed-Precision Perturbations

#### IEEE 754 Hardware Specifications

| Format | Exponent Bits | Mantissa (Fraction) Bits | Dynamic Range ($V_{\max}$) | Precision ($\epsilon_{\text{mach}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** (Single) | 8 | 23 | $\approx 3.4 \times 10^{38}$ | $\approx 1.19 \times 10^{-7}$ |
| **BF16** (Brain Float) | 8 | 7 | $\approx 3.4 \times 10^{38}$ | $\approx 7.81 \times 10^{-3}$ |
| **FP16** (Half) | 5 | 10 | **$65,504$** | $\approx 9.77 \times 10^{-4}$ |

#### Why Dot-Products Explode in Half-Precision
Let $q, k \in \mathbb{R}^d$ be zero-mean vectors with unit variance $\text{Var}(q_i) = \text{Var}(k_i) = 1$. The dot product is:
$$z = q^\top k = \sum_{i=1}^d q_i k_i$$
Under independence, the expectation and variance are:
$$\mathbb{E}[z] = 0, \quad \text{Var}(z) = \sum_{i=1}^d \text{Var}(q_i k_i) = d$$
For a standard hidden dimension $d = 128$, the standard deviation is $\sigma = \sqrt{128} \approx 11.3$. Typical logit values reach $\pm 3\sigma \approx \pm 34$.

In FP16:
$$e^{34} \approx 5.8 \times 10^{14} \gg 65,504$$
The exponential operation in $\text{softmax}(q^\top k)$ instantly overflows to $+\infty$, which produces $\text{NaN}$ in the normalization step $\frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{\infty}{\infty} = \text{NaN}$.

#### Relative Frobenius Norm Perturbation
`syndset` measures numerical sensitivity between FP32 execution $Y_{\text{fp32}}$ and half-precision execution $Y_{\text{half}}$ using the relative Frobenius norm:
$$\delta_{\text{rel}} = \frac{\|Y_{\text{half}} - Y_{\text{fp32}}\|_F}{\|Y_{\text{fp32}}\|_F + 10^{-7}}$$
where $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$. If $\delta_{\text{rel}} > 0.15$, internal variance reductions (e.g., LayerNorm variance calculation $\frac{1}{d} \sum (x_i - \mu)^2$) suffer from catastrophic cancellation under reduced mantissa bits and should remain in FP32.

---

### 6. Synthetic Task Theory & Inductive Biases

#### Associative Recall & Key-Value Retrieval
* **Structure:** Sequence $S = (k_1, v_1, k_2, v_2, \dots, k_N, v_N, q)$, where $q = k_\tau$, target $y = v_\tau$.
* **Attention Mechanism:** Standard softmax self-attention solves this in $\mathcal{O}(1)$ depth. When $Q_q$ aligns with $K_\tau$, the softmax weight focuses on $V_\tau$, routing the target directly to the output.
* **Recurrent / State-Space Model (SSM) Bound:** A fixed-size recurrent network updates state $h_t \in \mathbb{R}^D$ via:
  $$h_t = A h_{t-1} + B x_t$$
  Storing $N$ key-value pairs with vocabulary size $|\mathcal{V}|$ requires memorizing at least $N \log_2 |\mathcal{V}_{\text{val}}|$ bits of Shannon entropy. Linear Time-Invariant (LTI) systems compress past inputs with constant decay matrices, causing catastrophic forgetting as $N$ grows. This benchmark verifies whether an architecture incorporates input-dependent selection mechanisms (such as Mamba's input-dependent $\Delta_t, B_t, C_t$) to filter and retain discrete associations.

#### Induction Heads ($A \dots B \dots A \to B$)
* **Mechanism:** Discovered by Anthropic (Elhage et al., 2021), induction heads are the fundamental 2-layer attention subcircuit responsible for in-context learning in large language models.
* **Circuit Operation:**
  1. **Layer 1:** Attends to the previous token ($B$ attends to $A$).
  2. **Layer 2:** Attends to the position whose previous token matches current token $A$, copying token $B$ to the current prediction.
* **What It Tests:** Confirms whether a sequence architecture has the compositionality required to perform in-context associative pattern replication.

#### Selective Copying
* **Structure:** Signal tokens are interspersed among meaningless noise/distractor tokens. The model must output only the signal tokens at the end of the sequence.
* **What It Tests:** Differentiates content-aware architectures from static convolutional or stationary filtering models. Linear time-invariant filters cannot change their impulse response based on token values; this test verifies that the model's gating can suppress noise dynamically.

#### Cumulative Parity
* **Structure:** Given a binary sequence $x \in \{0, 1\}^T$, the target at step $t$ is $y_t = \left(\sum_{i=1}^t x_i\right) \pmod 2$.
* **What It Tests:** Parity is non-linearly separable and requires discrete state transitions across long horizons. Feedforward networks without recurrence or attention fail to maintain parity as $T$ scales; recurrent architectures must maintain unitary or orthogonal state transitions to avoid state degradation.

#### Spectral Bias & Harmonic Superposition
* **The Frequency Principle (F-Principle):**
  Rahaman et al. (2019) and Xu et al. (2019) demonstrated that neural networks trained via gradient descent fit target functions from **low to high frequencies**.
  For target signal $f(t) = \sum_k A_k \sin(2\pi \omega_k t + \phi_k)$, the convergence rate of Fourier mode $\omega$ decays rapidly with frequency:
  $$\frac{d}{dt} |\hat{f}(\omega) - \hat{f}_{\text{target}}(\omega)|^2 \propto -\lambda(\omega) |\hat{f}(\omega) - \hat{f}_{\text{target}}(\omega)|^2, \quad \lambda(\omega) \sim \mathcal{O}\left(\frac{1}{\omega^{2d}}\right)$$
* **What It Tests:** If a model cannot fit higher harmonics ($\omega \ge 6$) within initial optimization steps, it suffers from severe spectral bias. This identifies that the architecture needs Fourier feature mappings, sinusoidal position embeddings, or multi-scale convolutional kernels.

#### Concentric Hyperspheres & Manifold Curvature
* **Structure:** Points in $\mathbb{R}^D$ partitioned into nested spherical shells:
  $$\mathcal{C}_k = \{x \in \mathbb{R}^D \mid r_{k-1} \le \|x\|_2 < r_k\}$$
* **Topological Property:** The convex hulls of nested shells overlap at the origin:
  $$\text{Conv}(\mathcal{C}_i) \cap \text{Conv}(\mathcal{C}_j) \neq \emptyset$$
  Consequently, **no linear hyperplane can separate the classes**.
* **Depth Efficiency:** By depth-separation theorems (Telgarsky, 2016; Eldan & Shamir, 2016), approximating radial boundaries using piecewise linear activations (ReLU) with a shallow 1-hidden-layer network requires $\Omega(2^{D/2})$ neurons, whereas a deep network with $L \ge 3$ layers can represent it with $\mathcal{O}(D)$ neurons. This test evaluates whether an architecture achieves depth efficiency on curved non-linear manifolds.

#### Ill-Conditioned Regression & Optimization Curvature
* **Structure:** Linear regression $y = X \beta + \epsilon$, where the feature covariance matrix $\Sigma = \frac{1}{N} X^\top X$ has eigenvalues that decay geometrically:
  $$\lambda_i = 10^{-\frac{i-1}{D-1} \log_{10}(\kappa)}$$
* **Hessian Condition Number:**
  The condition number of the quadratic loss Hessian is:
  $$\kappa = \frac{\lambda_{\max}(\Sigma)}{\lambda_{\min}(\Sigma)} = 10^4$$
* **Convergence Rate Bound:**
  The distance to optimal weights $w^*$ under gradient descent with optimal step size is bounded by:
  $$\|w^{(t)} - w^*\|_2 \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \|w^{(0)} - w^*\|_2$$
  When $\kappa = 10^4$:
  $$\frac{\kappa - 1}{\kappa + 1} = \frac{9999}{10001} \approx 0.9998$$
  Standard first-order updates oscillate across the narrow ravine walls. This benchmark evaluates whether architectural normalization layers (BatchNorm, LayerNorm, Pre-LN) successfully whiten internal feature distributions ($\Sigma \approx I$) to achieve pre-conditioned optimization ($\kappa \approx 1$).

---

## Development & Publishing

### Running Tests
```bash
pytest -v
```

### Code Formatting & Linting
```bash
ruff check .
ruff format --check .
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
