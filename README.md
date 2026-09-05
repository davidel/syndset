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

Every diagnostic probe attaches non-invasively to any standard PyTorch `nn.Module`. For complete mathematical derivations, symbol definitions, and failure regimes, see the [Mathematical Foundations & Theoretical Mechanics](#mathematical-foundations--theoretical-mechanics) section below.

### 1. Gradient Flow (`syn.check_gradient_flow`)
Hooks into backpropagation on a single step to verify that gradients reach every parameter.

```python
results = syn.check_gradient_flow(model, sample_input, target=sample_target)
print("Status:", results["status"])
print("Vanishing ratio:", results["vanishing_ratio"])
print("Unused parameters:", results["zero_grad_params"])
```

* **Detects:** Disconnected parameters ($\large \|\nabla_\theta \mathcal{L}\| = 0$), exponential gradient vanishing ($\large R_{\text{vanish}} < 10^{-5}$), and explosive gradient growth.
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

### 6. Weight Initialization Scaling & Spectral Radius (`syn.check_initialization_scale`)
Examines raw parameter distributions without passing data, comparing variances against Kaiming/Xavier bounds and computing spectral radius $\rho(W)$.

```python
init_info = syn.check_initialization_scale(model)
print("Status:", init_info["status"])
print("Spectral radii of square weights:", init_info["spectral_radii"])
```

* **Detects:** Miscalibrated weight scales causing exploding/vanishing activations, and square recurrent/residual matrices with divergent spectral radius ($\large \rho(W) > 1$).
* [Jump to Initialization & Spectral Radius Math &rarr;](#6-weight-initialization-scaling--spectral-radius)

### 7. Loss Curvature & Sharpness (`syn.check_curvature_sharpness`)
Estimates Hessian trace $\text{Tr}(H)$ and mean loss curvature in 5 fast backward passes using Hutchinson's randomized estimator.

```python
curv_info = syn.check_curvature_sharpness(model, sample_input, sample_target)
print("Hessian trace estimate:", curv_info["hessian_trace"])
print("Mean curvature (eigenvalue):", curv_info["mean_curvature"])
```

* **Detects:** Hyper-sharp loss ravines prone to optimization instability and poor generalization.
* [Jump to Curvature & Hutchinson Trace Math &rarr;](#7-loss-surface-curvature--hutchinsons-hessian-trace-estimator)

### 8. Permutation Equivariance & Invariance (`syn.check_permutation_equivariance`)
Verifies whether set architectures (DeepSets, Set Transformers) or GNNs satisfy coordinate permutation symmetries along a given axis.

```python
perm_info = syn.check_permutation_equivariance(model, sample_input, perm_dim=1)
print("Equivariant?", perm_info["is_satisfied"])
print("Relative error:", perm_info["relative_difference"])
```

* **Detects:** Unintended positional dependencies or asymmetric operations that violate permutation symmetry.
* [Jump to Permutation Equivariance Math &rarr;](#8-permutation-equivariance--invariance)

---

## Synthetic Benchmark Datasets Overview

`syndset` provides procedural datasets designed to test specific algorithmic and geometric capabilities without external downloads. Detailed theoretical motivations for each task are provided in the [Synthetic Task Theory & Inductive Biases](#9-synthetic-task-theory--inductive-biases) section below.

### LLM & Sequence Models (`syndset.domains.llm`)

| Dataset | What It Evaluates |
| :--- | :--- |
| `AssociativeRecallDataset` | Key-value associative binding and single retrieval (`[k1, v1, ..., kq] -> vq`). [Details &rarr;](#associative-recall--key-value-retrieval) |
| `MultiQueryAssociativeRecallDataset` | Multi-Query Associative Recall (MQAR), querying multiple distinct keys to test memory bounds. [Details &rarr;](#multi-query-associative-recall-mqar) |
| `InductionDataset` | In-context pattern completion (`[A, B, ..., A] -> B`), testing induction head mechanics. [Details &rarr;](#induction-heads-a--b--a--b) |
| `SelectiveCopyDataset` | Distractor noise filtering and ordered memory reproduction. [Details &rarr;](#selective-copying) |
| `CumulativeParityDataset` | Step-by-step cumulative XOR tracking to verify discrete state compositionality. [Details &rarr;](#cumulative-parity) |
| `DyckLanguageDataset` | Well-nested bracket language (Dyck-k) evaluating pushdown automaton stack memory. [Details &rarr;](#dyck-bracket-languages-pushdown-stack-memory) |
| `MarkovLanguageDataset` | Autoregressive statistical distribution modeling and causal leakage detection ($k$-th order Markov source). [Details &rarr;](#markov-language-process-statistical-distribution-modeling) |

```python
from syndset.domains.llm import MarkovLanguageDataset, MultiQueryAssociativeRecallDataset

# 1000 sequences, 8 key-value pairs, 3 queries at end, vocab size 64
data = MultiQueryAssociativeRecallDataset(num_samples=1000, num_pairs=8, num_queries=3)
tokens, targets = data[0]

# 1000 sequences, order-2 Markov process, 8-token alphabet with Dirichlet transitions
markov_data = MarkovLanguageDataset(num_samples=1000, seq_len=32, vocab_size=8, order=2)
tokens, next_tokens = markov_data[0]
print(f"Bayes-optimal entropy floor: {markov_data.theoretical_entropy:.3f} nats")
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
| `TwoSpiralsDataset` | Interlocking Archimedean spirals testing non-linear topological manifold untangling. [Details &rarr;](#two-spirals-topological-manifold) |

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

This section provides the exact mathematical derivations, symbol definitions, condition numbers, and empirical citations underlying each diagnostic probe and benchmark task in `syndset`.

---

### 1. Effective Rank & Dimensional Collapse

#### The Problem: Anisotropy and Representation Collapse
In deep architectures—especially Transformers lacking pre-layer normalization or contrastive networks lacking repulsive loss terms—hidden representations frequently suffer from **anisotropy** or **dimensional collapse** (Dong et al., 2021, *"Attention is not all you need"*). In this degenerate state, the network projects all $D$-dimensional hidden states into a tiny subspace (e.g., a 1D line or narrow cone in $\mathbb{R}^D$), rendering the vast majority of parameters useless.

#### SVD Representation
Let $H \in \mathbb{R}^{N \times D}$ denote the matrix of activations from a hidden layer across $N$ samples (or $N = B \times T$ token representations in sequence models), with hidden dimension $D$.

We compute the Singular Value Decomposition (SVD):

$$
\large
H = U \Sigma V^\top
$$

**Symbol Definitions:**
* $H \in \mathbb{R}^{N \times D}$: The hidden activation matrix, where row $i$ represents the activation vector of sample (or token) $i$.
* $N \in \mathbb{N}$: The number of samples in the batch (or total token count $B \times T$ for sequences).
* $D \in \mathbb{N}$: The feature dimension (number of hidden channels or embedding width).
* $U \in \mathbb{R}^{N \times N}$: The left singular orthogonal matrix ($U^\top U = I_N$), representing the sample-space orthonormal basis.
* $\Sigma \in \mathbb{R}^{N \times D}$: The diagonal singular value matrix containing non-negative values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$, where $r = \min(N, D)$.
* $V \in \mathbb{R}^{D \times D}$: The right singular orthogonal matrix ($V^\top V = I_D$), representing the feature-space orthonormal basis.
* $\sigma_i \in \mathbb{R}_{\ge 0}$: The $i$-th singular value, proportional to the standard deviation of activations along the $i$-th principal component axis.
* $r = \min(N, D)$: The maximum possible theoretical rank of matrix $H$.

Standard algebraic rank counts non-zero singular values: $\text{rank}(H) = \sum_{i=1}^r \mathbf{1}_{\{\sigma_i > 0\}}$. However, standard rank is an unreliable, discontinuous step function in numerical computing because floating-point precision noise introduces tiny non-zero singular values ($\sim 10^{-16}$).

#### Spectral Entropy Effective Rank Formulation
To measure genuine continuous dimensionality, `syndset` implements **Spectral Entropy Effective Rank** (Roy & Vetterli, 2007):

$$
\large
p_i = \frac{\sigma_i}{\sum_{j=1}^r \sigma_j}
$$

$$
\large
H(p) = -\sum_{i=1}^r p_i \ln(p_i)
$$

$$
\large
\text{erank}(H) = \exp\left(H(p)\right)
$$

**Symbol Definitions:**
* $p_i \in [0, 1]$: The normalized energy ratio of the $i$-th singular mode. Because $\sum_{i=1}^r p_i = 1$, the vector $p = (p_1, \dots, p_r)$ forms a valid discrete probability distribution.
* $H(p) \in [0, \ln r]$: The Shannon spectral entropy of the singular value distribution, measured in nats.
* $\ln$: The natural logarithm (base $e$).
* $\exp$: The exponential function ($e^x$).
* $\text{erank}(H) \in [1, r]$: The continuous effective rank of matrix $H$.

#### Theoretical Bounds & Interpretation
**Maximum Dimensionality (Isotropic Dispersion):**  
If all singular values are equal ($\sigma_1 = \sigma_2 = \dots = \sigma_r$), then $p_i = 1/r$ for all $i$:

$$
\large
H(p) = -\sum_{i=1}^r \frac{1}{r} \ln\left(\frac{1}{r}\right) = \ln(r) \implies \text{erank}(H) = \exp(\ln r) = r = \min(N, D)
$$

The layer utilizes 100% of its available representation space.

**Total Dimensional Collapse:**  
If a single singular value dominates all others ($\sigma_1 \gg \sigma_2 \approx 0$), then $p_1 \to 1$ and $p_{i>1} \to 0$:

$$
\large
H(p) \to 0 \implies \text{erank}(H) \to \exp(0) = 1
$$

Even if $D = 4096$, the entire batch is confined to a 1-dimensional line.

`syndset` monitors the **rank utilization ratio**:

$$
\large
\rho_{\text{rank}} = \frac{\text{erank}(H)}{\min(N, D)}
$$

**Symbol Definitions:**
* $\rho_{\text{rank}} \in [0, 1]$: The fraction of theoretical rank capacity utilized.
Layers with $\rho_{\text{rank}} < 0.05$ (less than 5% capacity utilization) trigger an architectural warning for dimensional collapse.

---

### 2. Gradient Flow & Jacobian Conditioning

#### The Multivariable Chain Rule
Consider an $L$-layer neural network parameterized by weight tensors $\{\theta_1, \dots, \theta_L\}$:

$$
\large
h_l = f_l(h_{l-1}, \theta_l), \quad l = 1, \dots, L
$$

**Symbol Definitions:**
* $L \in \mathbb{N}$: The total number of sequential layers in the network.
* $l \in \{1, \dots, L\}$: The layer index.
* $h_l \in \mathbb{R}^{d_l}$: The activation vector output by layer $l$, where $d_l$ is the dimensionality of layer $l$.
* $h_0 = x$: The input vector to the network.
* $h_L = \hat{y}$: The output prediction vector of the network.
* $\theta_l$: The trainable parameter tensor (weights and biases) of layer $l$.
* $f_l$: The forward transformation function of layer $l$ (linear mapping + non-linearity).
* $\mathcal{L}(h_L, y) \in \mathbb{R}$: The scalar loss function comparing prediction $h_L$ against ground-truth $y$.

By the multivariable chain rule, the backpropagated gradient with respect to the input of layer $l$ is:

$$
\large
\frac{\partial \mathcal{L}}{\partial h_{l-1}} = \frac{\partial \mathcal{L}}{\partial h_l} J_l
$$

where $J_l \in \mathbb{R}^{d_l \times d_{l-1}}$ is the layer Jacobian matrix:

$$
\large
[J_l]_{jk} = \frac{\partial [h_l]_j}{\partial [h_{l-1}]_k}
$$

**Symbol Definitions:**
* $J_l \in \mathbb{R}^{d_l \times d_{l-1}}$: The Jacobian matrix of layer $l$ evaluated at input $h_{l-1}$.
* $[J_l]_{jk}$: The partial derivative of the $j$-th output component of layer $l$ with respect to the $k$-th input component.

Iterating the chain rule backwards from the loss at layer $L$ to the input layer $h_0$:

$$
\large
\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^L J_l
$$

Applying the submultiplicative property of induced matrix 2-norms (spectral norm, equivalent to the maximum singular value $\sigma_{\max}$):

$$
\large
\left\|\frac{\partial \mathcal{L}}{\partial h_0}\right\|_2 \le \left\|\frac{\partial \mathcal{L}}{\partial h_L}\right\|_2 \prod_{l=1}^L \|J_l\|_2
$$

**Symbol Definitions:**
* $\|\cdot\|_2$: The Euclidean vector norm or induced matrix spectral norm ($\|A\|_2 = \sigma_{\max}(A)$).
* $\prod_{l=1}^L \|J_l\|_2$: The product of the spectral norms of all layer Jacobians along the backward path.

#### The Failure Regimes
**1. Exponential Gradient Vanishing ($\|J_l\|_2 \le 1 - \epsilon$):**  
If the spectral norm of typical layer Jacobians is bounded strictly below unity by $\epsilon > 0$:

$$
\large
\left\|\frac{\partial \mathcal{L}}{\partial h_0}\right\|_2 \le \mathcal{O}\left((1 - \epsilon)^L\right) \xrightarrow{L \to \infty} 0
$$

The bottom layers receive virtually zero gradient update, leaving early representations stuck at initialization. `syndset` computes the **vanishing ratio**:

$$
\large
R_{\text{vanish}} = \frac{\min_{l} \|\nabla_{\theta_l} \mathcal{L}\|_2}{\max_{l} \|\nabla_{\theta_l} \mathcal{L}\|_2 + 10^{-12}}
$$

   **Symbol Definitions:**
   * $R_{\text{vanish}} \in [0, 1]$: The ratio between the weakest layer parameter gradient norm and the strongest layer parameter gradient norm.
   * $\nabla_{\theta_l} \mathcal{L}$: The gradient tensor of the scalar loss with respect to parameter tensor $\theta_l$.
   * $10^{-12}$: Small epsilon preventing division by zero.
   If $R_{\text{vanish}} < 10^{-5}$, an architectural gradient vanishing failure is flagged.

2. **Exponential Gradient Explosion ($\|J_l\|_2 \ge 1 + \epsilon$):**
   If $\|J_l\|_2 \ge 1 + \epsilon$, the gradient norm grows as $\mathcal{O}((1+\epsilon)^L) \to \infty$, causing optimizer parameter divergence and numerical overflow (`inf`/`nan`).

3. **Disconnected Subgraphs & Dead Parameters:**
   If a trainable parameter tensor $\theta_k$ receives an exact gradient norm of $\|\nabla_{\theta_k} \mathcal{L}\|_2 = 0$ while subsequent layers receive non-zero gradients, it identifies a structural graph disconnect (e.g., an unintended `.detach()`, a misplaced return statement, or an unused conditional branch).

---

### 3. Gradient Signal-to-Noise Ratio (SNR)

In minibatch gradient descent, the parameter gradient evaluated on minibatch $B_k$ is a stochastic estimator $g_k = \nabla_\theta \mathcal{L}(B_k)$ of the true population gradient $\mathbb{E}[\nabla_\theta \mathcal{L}]$.

Over $K$ distinct minibatches, `syndset` calculates the sample mean vector and coordinate variance:

$$
\large
\bar{g} = \frac{1}{K} \sum_{k=1}^K g_k
$$

$$
\large
s_i^2 = \frac{1}{K - 1} \sum_{k=1}^K (g_{k, i} - \bar{g}_i)^2
$$

$$
\large
\text{SNR}(\theta) = \frac{\|\bar{g}\|_2}{\sqrt{\sum_{i=1}^P s_i^2} + 10^{-10}}
$$

**Symbol Definitions:**
* $K \in \mathbb{N}$: The total number of observed minibatches ($K \ge 2$).
* $g_k \in \mathbb{R}^P$: The gradient vector evaluated on minibatch $B_k$, where $P$ is the number of parameters in $\theta$.
* $g_{k, i} \in \mathbb{R}$: The gradient with respect to parameter coordinate $i$ on minibatch $k$.
* $\bar{g} \in \mathbb{R}^P$: The empirical sample mean gradient vector across the $K$ minibatches.
* $s_i^2 \in \mathbb{R}_{\ge 0}$: The empirical sample variance of the gradient along coordinate $i$.
* $P \in \mathbb{N}$: Total number of scalar parameter coordinates in tensor $\theta$.
* $\text{SNR}(\theta) \in \mathbb{R}_{\ge 0}$: The Signal-to-Noise Ratio of parameter tensor $\theta$.

#### Regimes
* **$\large \text{SNR} > 1.0$ (Signal-Dominated):** Minibatches agree on a consistent descent direction.
* **$\large \text{SNR} < 0.1$ (Noise-Dominated):** Gradient updates are dominated by stochastic variance rather than systematic descent, indicating that the chosen batch size is inadequate or the loss surface has high variance.

---

### 4. Dead Units & Activation Saturation

For a feature channel or neuron $j$, let $h_{b, j}$ denote its scalar activation for sample $b$ in a batch of size $B$.

The sample mean and sample variance across the batch are:

$$
\large
\bar{h}_j = \frac{1}{B} \sum_{b=1}^B h_{b, j}
$$

$$
\large
\text{Var}_B(h_j) = \frac{1}{B - 1} \sum_{b=1}^B (h_{b, j} - \bar{h}_j)^2
$$

**Symbol Definitions:**
* $B \in \mathbb{N}$: The batch size.
* $j \in \{1, \dots, D\}$: The neuron or feature channel index within the layer.
* $h_{b, j} \in \mathbb{R}$: The activation value of neuron $j$ for sample $b$.
* $\bar{h}_j \in \mathbb{R}$: The batch mean activation of neuron $j$.
* $\text{Var}_B(h_j) \in \mathbb{R}_{\ge 0}$: The sample variance of neuron $j$ across the batch.

A unit is classified as **dead** if $\text{Var}_B(h_j) < 10^{-8}$.

#### The Dying ReLU Phenomenon
For activation function $f(z) = \max(0, z)$, the derivative (subgradient) is:

$$
\large
f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}
$$

**Symbol Definitions:**
* $z \in \mathbb{R}$: The pre-activation scalar value ($z = w^\top x + b$).
* $f'(z) \in \{0, 1\}$: The local derivative of the ReLU activation.

If a neuron's weights are updated such that its pre-activation $z_{b, j} \le 0$ for all samples $b \in B$, then:

$$
\large
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \sum_{b=1}^B \frac{\partial \mathcal{L}}{\partial h_{b, j}} \cdot f'(z_{b, j}) \cdot x_{b, i} = 0
$$

**Symbol Definitions:**
* $w_{ij} \in \mathbb{R}$: The weight connecting input coordinate $i$ to neuron $j$.
* $x_{b, i} \in \mathbb{R}$: The $i$-th input feature for sample $b$.
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
Let $q, k \in \mathbb{R}^d$ be zero-mean query and key vectors with unit variance $\text{Var}(q_i) = \text{Var}(k_i) = 1$. The inner product logit is:

$$
\large
z = q^\top k = \sum_{i=1}^d q_i k_i
$$

Under independent coordinates, the expectation and variance are:

$$
\large
\mathbb{E}[z] = 0, \quad \text{Var}(z) = \sum_{i=1}^d \text{Var}(q_i k_i) = d
$$

**Symbol Definitions:**
* $q \in \mathbb{R}^d$: The query vector of dimension $d$.
* $k \in \mathbb{R}^d$: The key vector of dimension $d$.
* $d \in \mathbb{N}$: The head projection dimension (typically $64$ or $128$).
* $z \in \mathbb{R}$: The unscaled attention logit.
* $\text{Var}(z) = d$: The logit variance grows linearly with the head dimension $d$.

For a standard hidden dimension $d = 128$, the standard deviation is $\sigma = \sqrt{128} \approx 11.3$. Under a standard normal distribution, logits routinely reach values of $\pm 3\sigma \approx \pm 34$.

In FP16:

$$
\large
e^{34} \approx 5.8 \times 10^{14} \gg 65,504
$$

The exponential in $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ instantly overflows the maximum FP16 representable value ($65,504$) to $+\infty$, producing $\frac{\infty}{\infty} = \text{NaN}$.

#### Relative Frobenius Norm Perturbation
`syndset` measures numerical sensitivity between FP32 reference output $Y_{\text{fp32}}$ and half-precision output $Y_{\text{half}}$ using the relative Frobenius norm:

$$
\large
\delta_{\text{rel}} = \frac{\|Y_{\text{half}} - Y_{\text{fp32}}\|_F}{\|Y_{\text{fp32}}\|_F + 10^{-7}}
$$

$$
\large
\|A\|_F = \sqrt{\sum_{i=1}^M \sum_{j=1}^N A_{ij}^2}
$$

**Symbol Definitions:**
* $Y_{\text{fp32}} \in \mathbb{R}^{M \times N}$: The reference output tensor computed under full 32-bit floating point precision.
* $Y_{\text{half}} \in \mathbb{R}^{M \times N}$: The output tensor computed under half precision (bfloat16 or float16) cast back to float.
* $\|A\|_F \in \mathbb{R}_{\ge 0}$: The Frobenius matrix norm of tensor $A$.
* $\delta_{\text{rel}} \in \mathbb{R}_{\ge 0}$: The relative error perturbation.
If $\delta_{\text{rel}} > 0.15$, internal operations suffer from severe precision loss and should be kept in FP32.

---

### 6. Weight Initialization Scaling & Spectral Radius

#### Variance Envelopes
Proper initialization preserves activation and gradient variance across depth. For a layer with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs:

**Kaiming Normal (He et al., 2015):**

$$
\large
\sigma_{\text{kaiming}} = \sqrt{\frac{2}{d_{\text{in}}}}
$$

**Xavier / Glorot Normal (Glorot & Bengio, 2010):**

$$
\large
\sigma_{\text{xavier}} = \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}}
$$

**Symbol Definitions:**
* $d_{\text{in}} \in \mathbb{N}$: Fan-in (number of input channels multiplied by spatial receptive field size).
* $d_{\text{out}} \in \mathbb{N}$: Fan-out (number of output channels multiplied by spatial receptive field size).
* $\sigma_{\text{kaiming}} \in \mathbb{R}_{>0}$: Theoretical standard deviation that prevents signal vanishing under ReLU activations.
* $\sigma_{\text{xavier}} \in \mathbb{R}_{>0}$: Theoretical standard deviation that preserves variance for linear/tanh activations.

`syndset` computes the empirical standard deviation $\sigma_{\text{emp}} = \text{std}(W)$ and flags layers where:

$$
\large
\frac{\sigma_{\text{emp}}}{\sigma_{\text{kaiming}}} > 10.0 \quad \text{or} \quad \frac{\sigma_{\text{emp}}}{\sigma_{\text{kaiming}}} < 0.10
$$

#### Spectral Radius for Square Weight Matrices
For a square linear, recurrent, or residual transition matrix $W \in \mathbb{R}^{d \times d}$:

$$
\large
\rho(W) = \max_{i} |\lambda_i(W)|
$$

**Symbol Definitions:**
* $W \in \mathbb{R}^{d \times d}$: The square weight matrix.
* $\lambda_i(W) \in \mathbb{C}$: The $i$-th eigenvalue of $W$.
* $\rho(W) \in \mathbb{R}_{\ge 0}$: The spectral radius (maximum eigenvalue modulus).

**Dynamical Implication:** In un-normalized recurrent loops $h_t = W h_{t-1} = W^t h_0$, the state magnitude evolves as $\|h_t\| \sim \mathcal{O}(\rho(W)^t)$. If $\rho(W) > 1.0$, states explode exponentially; if $\rho(W) \ll 1.0$, past information is wiped out. `syndset` flags square matrices with $\rho(W) > 1.5$.

---

### 7. Loss Surface Curvature & Hutchinson's Hessian Trace Estimator

#### The Curvature Formulation
Let $\mathcal{L}(\theta)$ be the scalar loss evaluated at parameter vector $\theta \in \mathbb{R}^P$. The local curvature is described by the Hessian matrix:

$$
\large
H = \nabla_\theta^2 \mathcal{L}(\theta) \in \mathbb{R}^{P \times P}, \quad [H]_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}
$$

Because instantiating the full $P \times P$ matrix requires $\mathcal{O}(P^2)$ memory (petabytes for modern architectures), `syndset` implements **Hutchinson's randomized trace estimator** (Hutchinson, 1989):

$$
\large
\text{Tr}(H) = \mathbb{E}_{v \sim \mathcal{D}} \left[ v^\top H v \right] \approx \frac{1}{M} \sum_{m=1}^M v_m^\top H v_m
$$

where each random vector $v_m \in \{-1, +1\}^P$ is sampled from an independent Rademacher distribution.

#### Pearlmutter's Vector-Hessian Product
The matrix-vector product $H v_m$ is computed without ever forming $H$ using the chain rule identity:

$$
\large
H v_m = \nabla_\theta \left( \nabla_\theta \mathcal{L}(\theta)^\top v_m \right)
$$

**Symbol Definitions:**
* $P \in \mathbb{N}$: Total number of trainable parameters in the model.
* $H \in \mathbb{R}^{P \times P}$: The Hessian curvature matrix.
* $\text{Tr}(H) = \sum_{i=1}^P \lambda_i(H)$: The trace of the Hessian (sum of all curvature eigenvalues).
* $M \in \mathbb{N}$: The number of stochastic projection samples (default: 5).
* $v_m \in \{-1, +1\}^P$: Random Rademacher probe vector.
* $\displaystyle \large \bar{\lambda} = \frac{\text{Tr}(H)}{P}$: The mean curvature eigenvalue.

**Empirical Interpretation:**
* **$\large \bar{\lambda} > 50.0$ (Hyper-Sharp Minima):** High curvature ravines where gradient descent oscillates violently, sensitive to learning rate and floating point rounding.
* **$\large \bar{\lambda} < -1.0$ (Negative Curvature):** The loss surface is locally concave, indicating the model is initialized near a saddle point.

---

### 8. Permutation Equivariance & Invariance

In geometric deep learning (Bronstein et al., 2021), architectures processing sets (DeepSets, Set Transformers) or graphs (GNNs) must respect symmetry under coordinate permutations $\pi \in \mathcal{S}_n$.

#### Mathematical Definitions

**Permutation Equivariance (Node / Token Representations):**

$$
\large
f(\pi(X)) = \pi(f(X))
$$

**Permutation Invariance (Graph / Set Summary Predictions):**

$$
\large
f(\pi(X)) = f(X)
$$

**Symbol Definitions:**
* $X \in \mathbb{R}^{B \times N \times D}$: Input tensor, where $N$ is the number of elements/nodes along permutation dimension `perm_dim`.
* $\pi \in \mathcal{S}_N$: A random permutation operator that reorders elements along dimension $N$.
* $f$: The forward function of the neural network.
* $\displaystyle \large \delta_{\text{perm}} = \frac{\|f(\pi(X)) - \hat{Y}_{\text{expected}}\|_F}{\|\hat{Y}_{\text{expected}}\|_F + 10^{-7}}$: The relative Frobenius error.

If $\large \delta_{\text{perm}} > 10^{-3}$, the model inadvertently relies on element ordering (e.g. via unintended positional embeddings or asymmetric pooling).

---

### 9. Synthetic Task Theory & Inductive Biases

#### Associative Recall & Key-Value Retrieval
**Sequence Construction:**

$$
\large
S = (k_1, v_1, k_2, v_2, \dots, k_N, v_N, q), \quad q = k_\tau, \quad \text{target } y = v_\tau
$$

**Symbol Definitions:**
* $N \in \mathbb{N}$: The number of distinct key-value pairs in the sequence.
* $k_i \in \mathcal{K}$: The $i$-th key token sampled from key vocabulary $\mathcal{K}$.
* $v_i \in \mathcal{V}_{\text{val}}$: The $i$-th value token sampled from value vocabulary $\mathcal{V}_{\text{val}}$, where $\mathcal{K} \cap \mathcal{V}_{\text{val}} = \emptyset$.
* $q \in \mathcal{K}$: The query token at the sequence end, identical to some earlier key $k_\tau$.
* $y = v_\tau \in \mathcal{V}_{\text{val}}$: The ground truth target token to be retrieved.

**Attention Routing:**  
Softmax self-attention solves this in $\mathcal{O}(1)$ depth because $Q_q K_\tau^\top$ produces a large logit, routing $V_\tau$ directly to the output:

$$
\large
A = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

**Recurrent / State-Space Model Bound:**
  A fixed-size recurrent network updates state $h_t \in \mathbb{R}^D$ via $h_t = A_t h_{t-1} + B_t x_t, y_t = C_t h_t$. Storing $N$ key-value pairs requires memorizing at least $N \log_2 |\mathcal{V}_{\text{val}}|$ bits of Shannon entropy. Linear Time-Invariant (LTI) systems compress past inputs with constant decay matrices, causing catastrophic forgetting as $N$ grows. This benchmark verifies whether an architecture incorporates input-dependent selection mechanisms (such as Mamba's input-dependent $\Delta_t, B_t, C_t$) to filter and retain discrete associations.

#### Multi-Query Associative Recall (MQAR)
**Sequence Construction:**

$$
\large
S = (k_1, v_1, \dots, k_N, v_N, q_1, q_2, \dots, q_M), \quad \text{targets } y = (v_{q_1}, v_{q_2}, \dots, v_{q_M})
$$

**Symbol Definitions:**
* $M \in \mathbb{N}$: The number of distinct query tokens at the sequence end ($M \le N$).
* $q_m \in \{k_1, \dots, k_N\}$: The $m$-th query token.
* $y \in \mathcal{V}_{\text{val}}^M$: The vector of expected target value tokens.

**Why MQAR is the Modern Gold Standard:** In single-query recall, a model only needs to retrieve one item. In MQAR, the model must maintain simultaneous access to multiple memories without mutual interference, exposing the strict capacity limit of fixed-size state representations compared to full $\mathcal{O}(T^2)$ KV-cache attention.

#### Induction Heads ($A \dots B \dots A \to B$)
* **Mechanism:** Discovered by Anthropic (Elhage et al., 2021), induction heads are the fundamental 2-layer attention subcircuit responsible for in-context learning in large language models.
* **Circuit Operation:**
  1. **Layer 1:** Attends to the previous token ($B$ attends to $A$).
  2. **Layer 2:** Attends to the position whose previous token matches current token $A$, copying token $B$ to the current prediction.
* **What It Tests:** Confirms whether a sequence architecture has the compositionality required to perform in-context associative pattern replication.

#### Dyck Bracket Languages (Pushdown Stack Memory)
**Context-Free Grammar:**  
Dyck-$k$ consists of well-nested bracket words generated by:

$$
\large
\mathcal{S} \to \epsilon \mid \mathcal{S} \, (_i \, \mathcal{S} \, )_i \, \mathcal{S}, \quad i \in \{1, \dots, k\}
$$

**Target Formulation:**  
At each sequence step $t$, the target is the closing bracket matching the top of the stack:

$$
\large
y_t = \text{Match}(\text{top}(\text{Stack}_t))
$$

**Symbol Definitions:**
* $k \in \mathbb{N}$: Number of bracket types (e.g. $k=2$ for round and square brackets).
* $\text{Stack}_t$: The unclosed bracket stack at step $t$.
* $y_t$: The required closing token matching the top of stack.

**Theoretical Implication:** By Chomsky hierarchy theorems, recognizing Dyck-$k$ for $k \ge 2$ requires a **Pushdown Automaton** with unbounded stack memory. Fixed-state RNNs and Transformers without recurrence or scratchpads fail when nesting depth exceeds memory capacity.

#### Selective Copying
* **Structure:** Signal tokens are interspersed among meaningless noise/distractor tokens. The model must output only the signal tokens at the end of the sequence.
* **What It Tests:** Differentiates content-aware architectures from static convolutional or stationary filtering models. Linear time-invariant filters cannot change their impulse response based on token values; this test verifies that the model's gating can suppress noise dynamically.

#### Cumulative Parity
**Structure:**  
Given a binary sequence $x \in \{0, 1\}^T$, the target at step $t$ is:

$$
\large
y_t = \left(\sum_{i=1}^t x_i\right) \pmod 2
$$

**Symbol Definitions:**
* $T \in \mathbb{N}$: The total sequence length.
* $x_i \in \{0, 1\}$: The binary input bit at step $i$.
* $y_t \in \{0, 1\}$: The cumulative parity bit at step $t$.

**What It Tests:** Parity is non-linearly separable and requires discrete state transitions across long horizons. Feedforward networks without recurrence or attention fail to maintain parity as $T$ scales; recurrent architectures must maintain unitary or orthogonal state transitions to avoid state degradation.

#### Markov Language Process (Statistical Distribution Modeling)
**Markov-k Process Formulation:**  
An order-$k$ stationary Markov source over a discrete vocabulary $\mathcal{V} = \{0, 1, \dots, V - 1\}$. By the Markov property, the transition probability distribution of next token $x_t$ given past sequence history depends strictly on the trailing $k$ tokens:

$$
\large
P(x_t \mid x_1, \dots, x_{t-1}) = P(x_t \mid x_{t-k}, \dots, x_{t-1}) = P(x_t \mid c_t)
$$

where $c_t = (x_{t-k}, \dots, x_{t-1}) \in \mathcal{V}^k$ is the historical context prefix.

**Transition Matrix via Dirichlet Prior:**  
For each of the $V^k$ distinct context states $c \in \mathcal{V}^k$, the conditional transition distribution $\mathbf{p}_c = P(\cdot \mid c) \in \Delta^{V-1}$ is sampled from a symmetric Dirichlet prior with concentration parameter $\alpha > 0$:

$$
\large
\mathbf{p}_c \sim \text{Dirichlet}(\alpha \mathbf{1}_V), \quad p_c(v) = \frac{y_{c, v}}{\sum_{j=0}^{V-1} y_{c, j}}, \quad y_{c, j} \sim \text{Gamma}(\alpha, 1)
$$

- $\alpha < 1.0$: Concentrated, sparse transitions with low conditional entropy.
- $\alpha = 1.0$: Uniform measure over the probability simplex $\Delta^{V-1}$.
- $\alpha > 1.0$: Smooth, diffuse transitions approaching maximum entropy.

**Theoretical Bayes-Optimal Shannon Entropy Floor:**  
For any sequence of length $T$, the minimum cross-entropy loss achievable by any strictly causal model is the conditional Shannon entropy rate:

$$
\large
H(P^\star) = \mathbb{E}_{c \sim \pi} \left[ -\sum_{v \in \mathcal{V}} P(v \mid c) \ln P(v \mid c) \right]
$$

where $\pi$ is the stationary distribution over context states. By Shannon's source coding theorem and the non-negativity of Kullback–Leibler divergence:

$$
\large
\mathbb{E}[\mathcal{L}_{\mathrm{CE}}(Q)] = H(P^\star) + D_{\mathrm{KL}}(P^\star \parallel Q) \ge H(P^\star)
$$

**Symbol Definitions:**
* $V \in \mathbb{N}_{\ge 2}$: Discrete vocabulary size ($|\mathcal{V}| = V$).
* $k \in \mathbb{N}_{\ge 1}$: Markov dependency order (memory depth horizon).
* $c \in \mathcal{V}^k$: Historical context prefix of length $k$.
* $\alpha \in \mathbb{R}_{>0}$: Dirichlet concentration parameter.
* $H(P^\star)$: Theoretical Bayes-optimal Shannon entropy in nats.
* $Q(x_t \mid x_{<t})$: Model predicted probability distribution $\text{softmax}(z_t)$.
* $\displaystyle \large D_{\mathrm{KL}}(P^\star \parallel Q) = \sum_{v \in \mathcal{V}} P^\star(v \mid c) \ln \frac{P^\star(v \mid c)}{Q(v \mid c)}$: Forward KL divergence.
* $\displaystyle \large \mathrm{TV}(P^\star, Q) = \frac{1}{2} \sum_{v \in \mathcal{V}} |P^\star(v \mid c) - Q(v \mid c)|$: Total Variation distance in $[0, 1]$.

* **What It Tests & Inductive Biases:**
  1. **Continuous Softmax Calibration:** Unlike deterministic tasks where targets have zero entropy, this benchmark evaluates whether an architecture's logits and softmax output can calibrate to soft categorical distributions without logit explosion or mode collapse.
  2. **Causal Leakage Detection (`[LEAK]`):** Because $H(P^\star)$ is known analytically, an empirical test loss strictly below the theoretical entropy ($\mathcal{L}_{\mathrm{model}} < H(P^\star) - \epsilon$) is a mathematical impossibility for any causal model. If observed, `syndset` instantly flags a lookahead bug in the model's causal attention mask or temporal slicing.
  3. **Sub-10ms Exhaustive Prefix Auditing:** By passing the complete Cartesian product $\mathcal{V}^k$ through the model in a single batch, the entire learned transition manifold is verified against ground truth via Total Variation distance and KL divergence without Monte Carlo sampling variance.

#### Two Spirals Topological Manifold
**Parametric Formulation:**  
Two continuous intertwined Archimedean spirals:

$$
\large
\text{Spiral 1 } (y = 0): \quad x_1 = r(\theta) \cos(\theta) + \epsilon, \quad x_2 = r(\theta) \sin(\theta) + \epsilon
$$

$$
\large
\text{Spiral 2 } (y = 1): \quad x_1 = -r(\theta) \cos(\theta) + \epsilon, \quad x_2 = -r(\theta) \sin(\theta) + \epsilon
$$

where radial distance grows linearly with angle:

$$
\large
r(\theta) = \frac{\theta}{2\pi \cdot \text{turns}}, \quad \theta \in [0, 2\pi \cdot \text{turns}]
$$

**Symbol Definitions:**
* $\theta \in \mathbb{R}_{\ge 0}$: The continuous angular parameter in radians.
* $\text{turns} \in \mathbb{R}_{>0}$: Total revolutions of each spiral (default: 2.5).
* $r(\theta) \in [0, 1]$: Normalized radius.
* $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$: Gaussian coordinate noise.

**Topological Difficulty:** The decision boundary winds around the origin multiple times. A single linear layer or shallow MLP cannot separate the spirals; it requires at least 3 hidden layers with sufficient non-linear activations to partition the continuous manifold into piecewise convex regions.

#### Spectral Bias & Harmonic Superposition
**The Frequency Principle (F-Principle):**  
Rahaman et al. (2019) and Xu et al. (2019) demonstrated that neural networks trained via gradient descent fit target functions from **low to high frequencies**.
For target signal:

$$
\large
f(t) = \sum_{k=1}^K A_k \sin(2\pi \omega_k t + \phi_k), \quad A_k = \frac{1}{\sqrt{k}}, \quad \omega_k = 2k
$$

**Symbol Definitions:**
* $t \in [0, 1]$: Continuous time parameter.
* $K \in \mathbb{N}$: Number of superimposed harmonic frequencies.
* $k \in \{1, \dots, K\}$: Harmonic mode index.
* $A_k = 1/\sqrt{k} \in \mathbb{R}$: Amplitude of the $k$-th harmonic component.
* $\omega_k = 2k \in \mathbb{R}$: Frequency of the $k$-th harmonic component.
* $\phi_k \in [0, 2\pi)$: Random initial phase offset.

The convergence rate of Fourier mode $\omega$ decays rapidly with frequency:

$$
\large
\frac{d}{dt} |\hat{f}(\omega) - \hat{f}_{\text{target}}(\omega)|^2 \propto -\lambda(\omega) |\hat{f}(\omega) - \hat{f}_{\text{target}}(\omega)|^2, \quad \lambda(\omega) \sim \mathcal{O}\left(\frac{1}{\omega^{2d}}\right)
$$

  **Symbol Definitions:**
  * $\hat{f}(\omega)$: The Fourier transform of the neural network's function approximation at frequency $\omega$.
  * $\hat{f}_{\text{target}}(\omega)$: The Fourier transform of the true target function.
  * $\lambda(\omega)$: The spectral convergence rate eigenvalue.
  * $d \in \mathbb{N}$: The input dimension.

* **What It Tests:** If a model cannot fit higher harmonics ($\omega \ge 6$) within initial optimization steps, it suffers from severe spectral bias. This identifies that the architecture needs Fourier feature mappings, sinusoidal position embeddings, or multi-scale convolutional kernels.

#### Concentric Hyperspheres & Manifold Curvature
**Structure:**  
Points in $\mathbb{R}^D$ partitioned into nested spherical shells:

$$
\large
\mathcal{C}_k = \{x \in \mathbb{R}^D \mid r_{k-1} \le \|x\|_2 < r_k\}
$$

**Symbol Definitions:**
* $D \in \mathbb{N}$: Feature space dimensionality.
* $k \in \{1, \dots, K\}$: Shell class index.
* $\mathcal{C}_k \subset \mathbb{R}^D$: The set of points belonging to class $k$.
* $r_k \in \mathbb{R}_{>0}$: The outer radius boundary of shell $k$, with $0 = r_0 < r_1 < \dots < r_K$.
* $\|x\|_2 = \sqrt{\sum_{i=1}^D x_i^2}$: The Euclidean distance from the origin.

**Topological Property:**  
The convex hulls of nested shells overlap at the origin:

$$
\large
\text{Conv}(\mathcal{C}_i) \cap \text{Conv}(\mathcal{C}_j) \neq \emptyset, \quad \forall i \neq j
$$

  Consequently, **no linear hyperplane can separate the classes**.

* **Depth Efficiency:** By depth-separation theorems (Telgarsky, 2016; Eldan & Shamir, 2016), approximating radial boundaries using piecewise linear activations (ReLU) with a shallow 1-hidden-layer network requires $\Omega(2^{D/2})$ neurons, whereas a deep network with $L \ge 3$ layers can represent it with $\mathcal{O}(D)$ neurons. This test evaluates whether an architecture achieves depth efficiency on curved non-linear manifolds.

#### Ill-Conditioned Regression & Optimization Curvature
**Structure:**  
Linear regression $y = X \beta + \epsilon$, where the feature covariance matrix $\Sigma = \frac{1}{N} X^\top X$ has eigenvalues that decay geometrically:

$$
\large
\lambda_i = 10^{-\frac{i-1}{D-1} \log_{10}(\kappa)}, \quad i = 1, \dots, D
$$

**Symbol Definitions:**
* $X \in \mathbb{R}^{N \times D}$: The feature design matrix.
* $\beta \in \mathbb{R}^D$: The true linear weight vector.
* $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$: Additive Gaussian observation noise.
* $\Sigma \in \mathbb{R}^{D \times D}$: The empirical feature covariance matrix.
* $\lambda_i \in \mathbb{R}_{>0}$: The $i$-th eigenvalue of covariance matrix $\Sigma$.
* $\kappa \in \mathbb{R}_{\ge 1}$: The target condition number.

**Hessian Condition Number:**  
The condition number of the quadratic loss Hessian is:

$$
\large
\kappa = \frac{\lambda_{\max}(\Sigma)}{\lambda_{\min}(\Sigma)} = 10^4
$$

**Symbol Definitions:**
* $\lambda_{\max}(\Sigma) = \lambda_1$: The largest eigenvalue of $\Sigma$.
* $\lambda_{\min}(\Sigma) = \lambda_D$: The smallest eigenvalue of $\Sigma$.
* $\kappa$: The condition number, measuring the ratio of maximum to minimum curvature of the loss ravine.

**Convergence Rate Bound:**  
The distance to optimal weights $w^\star$ under gradient descent with optimal step size is bounded by:

$$
\large
\|w^{(t)} - w^\star\|_2 \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^{t} \|w^{(0)} - w^\star\|_2
$$

**Symbol Definitions:**
* $w^{(t)} \in \mathbb{R}^D$: The parameter weight vector at gradient step $t$.
* $w^\star \in \mathbb{R}^D$: The optimal analytical least-squares solution.
* $t \in \mathbb{N}$: The iteration step number.
* $\displaystyle \large \frac{\kappa - 1}{\kappa + 1} \in [0, 1)$: The convergence contraction factor.

When $\kappa = 10^4$:

$$
\large
\frac{\kappa - 1}{\kappa + 1} = \frac{9999}{10001} \approx 0.9998
$$

  Standard first-order updates oscillate across the narrow ravine walls. This benchmark evaluates whether architectural normalization layers (BatchNorm, LayerNorm, Pre-LN) successfully whiten internal feature distributions ($\Sigma \approx I$) to achieve pre-conditioned optimization ($\kappa \approx 1$).

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
