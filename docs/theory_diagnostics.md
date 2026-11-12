# Mathematical Foundations: Diagnostic Probes & Decision Mechanics

This document provides the complete mathematical formulations, derivations, symbol definitions, condition bounds, and failure regimes for all diagnostic probes and multi-criteria decision mechanics in `syndset`.

---

### 1. Effective Rank & Dimensional Collapse

#### The Problem: Anisotropy and Representation Collapse
In deep architectures—especially Transformers lacking pre-layer normalization or contrastive networks lacking repulsive loss terms—hidden representations frequently suffer from **anisotropy** or **dimensional collapse** (Dong et al., 2021, *"Attention is not all you need"*). In this degenerate state, the network projects all $`D`$-dimensional hidden states into a tiny subspace (e.g., a 1D line or narrow cone in $`\mathbb{R}^D`$), rendering the vast majority of parameters useless.

#### SVD Representation
Let $`H \in \mathbb{R}^{N \times D}`$ denote the matrix of activations from a hidden layer across $`N`$ samples (or $`N = B \times T`$ token representations in sequence models), with hidden dimension $`D`$.

We compute the Singular Value Decomposition (SVD):

$$
\large
H = U \Sigma V^\top
$$

**Symbol Definitions:**
* $`H \in \mathbb{R}^{N \times D}`$: The hidden activation matrix, where row $`i`$ represents the activation vector of sample (or token) $`i`$.
* $`N \in \mathbb{N}`$: The number of samples in the batch (or total token count $`B \times T`$ for sequences).
* $`D \in \mathbb{N}`$: The feature dimension (number of hidden channels or embedding width).
* $`U \in \mathbb{R}^{N \times N}`$: The left singular orthogonal matrix ($`U^\top U = I_N`$), representing the sample-space orthonormal basis.
* $`\Sigma \in \mathbb{R}^{N \times D}`$: The diagonal singular value matrix containing non-negative values $`\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0`$, where $`r = \min(N, D)`$.
* $`V \in \mathbb{R}^{D \times D}`$: The right singular orthogonal matrix ($`V^\top V = I_D`$), representing the feature-space orthonormal basis.
* $`\sigma_i \in \mathbb{R}_{\ge 0}`$: The $`i`$-th singular value, proportional to the standard deviation of activations along the $`i`$-th principal component axis.
* $`r = \min(N, D)`$: The maximum possible theoretical rank of matrix $`H`$.

Standard algebraic rank counts non-zero singular values: $`\text{rank}(H) = \sum_{i=1}^r \mathbf{1}_{\{\sigma_i \gt 0\}}`$. However, standard rank is an unreliable, discontinuous step function in numerical computing because floating-point precision noise introduces tiny non-zero singular values ($`\sim 10^{-16}`$).

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
* $`p_i \in [0, 1]`$: The normalized energy ratio of the $`i`$-th singular mode. Because $`\sum_{i=1}^r p_i = 1`$, the vector $`p = (p_1, \dots, p_r)`$ forms a valid discrete probability distribution.
* $`H(p) \in [0, \ln r]`$: The Shannon spectral entropy of the singular value distribution, measured in nats.
* $`\ln`$: The natural logarithm (base $`e`$).
* $`\exp`$: The exponential function ($`e^x`$).
* $`\text{erank}(H) \in [1, r]`$: The continuous effective rank of matrix $`H`$.

#### Theoretical Bounds & Interpretation
**Maximum Dimensionality (Isotropic Dispersion):**  
If all singular values are equal ($`\sigma_1 = \sigma_2 = \dots = \sigma_r`$), then $`p_i = 1/r`$ for all $`i`$:

$$
\large
H(p) = -\sum_{i=1}^r \frac{1}{r} \ln\left(\frac{1}{r}\right) = \ln(r) \implies \text{erank}(H) = \exp(\ln r) = r = \min(N, D)
$$

The layer utilizes 100% of its available representation space.

**Total Dimensional Collapse:**  
If a single singular value dominates all others ($`\sigma_1 \gg \sigma_2 \approx 0`$), then $`p_1 \to 1`$ and $`p_{i\gt 1} \to 0`$:

$$
\large
H(p) \to 0 \implies \text{erank}(H) \to \exp(0) = 1
$$

Even if $`D = 4096`$, the entire batch is confined to a 1-dimensional line.

`syndset` monitors the **rank utilization ratio**:

$$
\large
\rho_{\text{rank}} = \frac{\text{erank}(H)}{\min(N, D)}
$$

**Symbol Definitions:**
* $`\rho_{\text{rank}} \in [0, 1]`$: The fraction of theoretical rank capacity utilized.
Layers with $`\rho_{\text{rank}} \lt 0.05`$ (less than 5% capacity utilization) trigger an architectural warning for dimensional collapse.

---

### 2. Gradient Flow & Jacobian Conditioning

#### The Multivariable Chain Rule
Consider an $`L`$-layer neural network parameterized by weight tensors $`\{\theta_1, \dots, \theta_L\}`$:

$$
\large
h_l = f_l(h_{l-1}, \theta_l), \quad l = 1, \dots, L
$$

**Symbol Definitions:**
* $`L \in \mathbb{N}`$: The total number of sequential layers in the network.
* $`l \in \{1, \dots, L\}`$: The layer index.
* $`h_l \in \mathbb{R}^{d_l}`$: The activation vector output by layer $`l`$, where $`d_l`$ is the dimensionality of layer $`l`$.
* $`h_0 = x`$: The input vector to the network.
* $`h_L = \hat{y}`$: The output prediction vector of the network.
* $`\theta_l`$: The trainable parameter tensor (weights and biases) of layer $`l`$.
* $`f_l`$: The forward transformation function of layer $`l`$ (linear mapping + non-linearity).
* $`\mathcal{L}(h_L, y) \in \mathbb{R}`$: The scalar loss function comparing prediction $`h_L`$ against ground-truth $`y`$.

By the multivariable chain rule, the backpropagated gradient with respect to the input of layer $`l`$ is:

$$
\large
\frac{\partial \mathcal{L}}{\partial h_{l-1}} = \frac{\partial \mathcal{L}}{\partial h_l} J_l
$$

where $`J_l \in \mathbb{R}^{d_l \times d_{l-1}}`$ is the layer Jacobian matrix:

$$
\large
[J_l]_{jk} = \frac{\partial [h_l]_j}{\partial [h_{l-1}]_k}
$$

**Symbol Definitions:**
* $`J_l \in \mathbb{R}^{d_l \times d_{l-1}}`$: The Jacobian matrix of layer $`l`$ evaluated at input $`h_{l-1}`$.
* $`[J_l]_{jk}`$: The partial derivative of the $`j`$-th output component of layer $`l`$ with respect to the $`k`$-th input component.

Iterating the chain rule backwards from the loss at layer $`L`$ to the input layer $`h_0`$:

$$
\large
\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^L J_l
$$

Applying the submultiplicative property of induced matrix 2-norms (spectral norm, equivalent to the maximum singular value $`\sigma_{\max}`$):

$$
\large
\left\\|\frac{\partial \mathcal{L}}{\partial h_0}\right\\|_2 \le \left\\|\frac{\partial \mathcal{L}}{\partial h_L}\right\\|_2 \prod_{l=1}^L \\|J_l\\|_2
$$

**Symbol Definitions:**
* $`\|\cdot\|_2`$: The Euclidean vector norm or induced matrix spectral norm ($`\|A\|_2 = \sigma_{\max}(A)`$).
* $`\prod_{l=1}^L \|J_l\|_2`$: The product of the spectral norms of all layer Jacobians along the backward path.

#### The Failure Regimes
**1. Exponential Gradient Vanishing ($`\|J_l\|_2 \le 1 - \epsilon`$):**  
If the spectral norm of typical layer Jacobians is bounded strictly below unity by $`\epsilon \gt 0`$:

$$
\large
\left\\|\frac{\partial \mathcal{L}}{\partial h_0}\right\\|_2 \le \mathcal{O}\left((1 - \epsilon)^L\right) \xrightarrow{L \to \infty} 0
$$

The bottom layers receive virtually zero gradient update, leaving early representations stuck at initialization. `syndset` computes the **vanishing ratio**:

$$
\large
R_{\text{vanish}} = \frac{\min_{l} \\|\nabla_{\theta_l} \mathcal{L}\\|_2}{\max_{l} \\|\nabla_{\theta_l} \mathcal{L}\\|_2 + 10^{-12}}
$$

   **Symbol Definitions:**
   * $`R_{\text{vanish}} \in [0, 1]`$: The ratio between the weakest layer parameter gradient norm and the strongest layer parameter gradient norm.
   * $`\nabla_{\theta_l} \mathcal{L}`$: The gradient tensor of the scalar loss with respect to parameter tensor $`\theta_l`$.
   * $`10^{-12}`$: Small epsilon preventing division by zero.
   If $`R_{\text{vanish}} \lt 10^{-5}`$, an architectural gradient vanishing failure is flagged.

2. **Exponential Gradient Explosion ($`\|J_l\|_2 \ge 1 + \epsilon`$):**
   If $`\|J_l\|_2 \ge 1 + \epsilon`$, the gradient norm grows as $`\mathcal{O}((1+\epsilon)^L) \to \infty`$, causing optimizer parameter divergence and numerical overflow (`inf`/`nan`).

3. **Disconnected Subgraphs & Dead Parameters:**
   If a trainable parameter tensor $`\theta_k`$ receives an exact gradient norm of $`\|\nabla_{\theta_k} \mathcal{L}\|_2 = 0`$ while subsequent layers receive non-zero gradients, it identifies a structural graph disconnect (e.g., an unintended `.detach()`, a misplaced return statement, or an unused conditional branch).

---

### 3. Gradient Signal-to-Noise Ratio (SNR)

In minibatch gradient descent, the parameter gradient evaluated on minibatch $`B_k`$ is a stochastic estimator $`g_k = \nabla_\theta \mathcal{L}(B_k)`$ of the true population gradient $`\mathbb{E}[\nabla_\theta \mathcal{L}]`$.

Over $`K`$ distinct minibatches, `syndset` calculates the sample mean vector and coordinate variance:

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
\text{SNR}(\theta) = \frac{\\|\bar{g}\\|_2}{\sqrt{\sum_{i=1}^P s_i^2} + 10^{-10}}
$$

**Symbol Definitions:**
* $`K \in \mathbb{N}`$: The total number of observed minibatches ($`K \ge 2`$).
* $`g_k \in \mathbb{R}^P`$: The gradient vector evaluated on minibatch $`B_k`$, where $`P`$ is the number of parameters in $`\theta`$.
* $`g_{k, i} \in \mathbb{R}`$: The gradient with respect to parameter coordinate $`i`$ on minibatch $`k`$.
* $`\bar{g} \in \mathbb{R}^P`$: The empirical sample mean gradient vector across the $`K`$ minibatches.
* $`s_i^2 \in \mathbb{R}_{\ge 0}`$: The empirical sample variance of the gradient along coordinate $`i`$.
* $`P \in \mathbb{N}`$: Total number of scalar parameter coordinates in tensor $`\theta`$.
* $`\text{SNR}(\theta) \in \mathbb{R}_{\ge 0}`$: The Signal-to-Noise Ratio of parameter tensor $`\theta`$.

#### Regimes
* **$`\large \text{SNR} \gt 1.0`$ (Signal-Dominated):** Minibatches agree on a consistent descent direction.
* **$`\large \text{SNR} \lt 0.1`$ (Noise-Dominated):** Gradient updates are dominated by stochastic variance rather than systematic descent, indicating that the chosen batch size is inadequate or the loss surface has high variance.

---

### 4. Dead Units & Activation Saturation

For a feature channel or neuron $`j`$, let $`h_{b, j}`$ denote its scalar activation for sample $`b`$ in a batch of size $`B`$.

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
* $`B \in \mathbb{N}`$: The batch size.
* $`j \in \{1, \dots, D\}`$: The neuron or feature channel index within the layer.
* $`h_{b, j} \in \mathbb{R}`$: The activation value of neuron $`j`$ for sample $`b`$.
* $`\bar{h}_j \in \mathbb{R}`$: The batch mean activation of neuron $`j`$.
* $`\text{Var}_B(h_j) \in \mathbb{R}_{\ge 0}`$: The sample variance of neuron $`j`$ across the batch.

A unit is classified as **dead** if $`\text{Var}_B(h_j) \lt 10^{-8}`$.

#### The Dying ReLU Phenomenon
For activation function $`f(z) = \max(0, z)`$, the derivative (subgradient) is:

$$
\large
f'(z) = \begin{cases} 1 & \text{if } z \gt 0 \\ 0 & \text{if } z \le 0 \end{cases}
$$

**Symbol Definitions:**
* $`z \in \mathbb{R}`$: The pre-activation scalar value ($`z = w^\top x + b`$).
* $`f'(z) \in \{0, 1\}`$: The local derivative of the ReLU activation.

If a neuron's weights are updated such that its pre-activation $`z_{b, j} \le 0`$ for all samples $`b \in B`$, then:

$$
\large
\frac{\partial \mathcal{L}}{\partial w_{ij}} = \sum_{b=1}^B \frac{\partial \mathcal{L}}{\partial h_{b, j}} \cdot f'(z_{b, j}) \cdot x_{b, i} = 0
$$

**Symbol Definitions:**
* $`w_{ij} \in \mathbb{R}`$: The weight connecting input coordinate $`i`$ to neuron $`j`$.
* $`x_{b, i} \in \mathbb{R}`$: The $`i`$-th input feature for sample $`b`$.
The neuron produces a constant output of 0 and receives a gradient of 0, making it permanently incapable of recovering. `syndset` flags any layer where more than 30% of units are dead.

---

### 5. Floating-Point Dynamic Range & Mixed-Precision Perturbations

#### IEEE 754 Hardware Specifications

| Format | Exponent Bits | Mantissa (Fraction) Bits | Dynamic Range ($`V_{\max}`$) | Precision ($`\epsilon_{\text{mach}}`$) |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** (Single) | 8 | 23 | $`\approx 3.4 \times 10^{38}`$ | $`\approx 1.19 \times 10^{-7}`$ |
| **BF16** (Brain Float) | 8 | 7 | $`\approx 3.4 \times 10^{38}`$ | $`\approx 7.81 \times 10^{-3}`$ |
| **FP16** (Half) | 5 | 10 | **$`65,504`$** | $`\approx 9.77 \times 10^{-4}`$ |

#### Why Dot-Products Explode in Half-Precision
Let $`q, k \in \mathbb{R}^d`$ be zero-mean query and key vectors with unit variance $`\text{Var}(q_i) = \text{Var}(k_i) = 1`$. The inner product logit is:

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
* $`q \in \mathbb{R}^d`$: The query vector of dimension $`d`$.
* $`k \in \mathbb{R}^d`$: The key vector of dimension $`d`$.
* $`d \in \mathbb{N}`$: The head projection dimension (typically $`64`$ or $`128`$).
* $`z \in \mathbb{R}`$: The unscaled attention logit.
* $`\text{Var}(z) = d`$: The logit variance grows linearly with the head dimension $`d`$.

For a standard hidden dimension $`d = 128`$, the standard deviation is $`\sigma = \sqrt{128} \approx 11.3`$. Under a standard normal distribution, logits routinely reach values of $`\pm 3\sigma \approx \pm 34`$.

In FP16:

$$
\large
e^{34} \approx 5.8 \times 10^{14} \gg 65,504
$$

The exponential in $`\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}`$ instantly overflows the maximum FP16 representable value ($`65,504`$) to $`+\infty`$, producing $`\frac{\infty}{\infty} = \text{NaN}`$.

#### Relative Frobenius Norm Perturbation
`syndset` measures numerical sensitivity between FP32 reference output $`Y_{\text{fp32}}`$ and half-precision output $`Y_{\text{half}}`$ using the relative Frobenius norm:

$$
\large
\delta_{\text{rel}} = \frac{\\|Y_{\text{half}} - Y_{\text{fp32}}\\|_F}{\\|Y_{\text{fp32}}\\|_F + 10^{-7}}
$$

$$
\large
\\|A\\|_F = \sqrt{\sum_{i=1}^M \sum_{j=1}^N A_{ij}^2}
$$

**Symbol Definitions:**
* $`Y_{\text{fp32}} \in \mathbb{R}^{M \times N}`$: The reference output tensor computed under full 32-bit floating point precision.
* $`Y_{\text{half}} \in \mathbb{R}^{M \times N}`$: The output tensor computed under half precision (bfloat16 or float16) cast back to float.
* $`\|A\|_F \in \mathbb{R}_{\ge 0}`$: The Frobenius matrix norm of tensor $`A`$.
* $`\delta_{\text{rel}} \in \mathbb{R}_{\ge 0}`$: The relative error perturbation.
If $`\delta_{\text{rel}} \gt 0.15`$, internal operations suffer from severe precision loss and should be kept in FP32.

---

### 6. Weight Initialization Scaling & Spectral Radius

#### Variance Envelopes
Proper initialization preserves activation and gradient variance across depth. For a layer with $`d_{\text{in}}`$ inputs and $`d_{\text{out}}`$ outputs:

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
* $`d_{\text{in}} \in \mathbb{N}`$: Fan-in (number of input channels multiplied by spatial receptive field size).
* $`d_{\text{out}} \in \mathbb{N}`$: Fan-out (number of output channels multiplied by spatial receptive field size).
* $`\sigma_{\text{kaiming}} \in \mathbb{R}_{\gt 0}`$: Theoretical standard deviation that prevents signal vanishing under ReLU activations.
* $`\sigma_{\text{xavier}} \in \mathbb{R}_{\gt 0}`$: Theoretical standard deviation that preserves variance for linear/tanh activations.

`syndset` computes the empirical standard deviation $`\sigma_{\text{emp}} = \text{std}(W)`$ and flags layers where:

$$
\large
\frac{\sigma_{\text{emp}}}{\sigma_{\text{kaiming}}} \gt 10.0 \quad \text{or} \quad \frac{\sigma_{\text{emp}}}{\sigma_{\text{kaiming}}} \lt 0.10
$$

#### Spectral Radius for Square Weight Matrices
For a square linear, recurrent, or residual transition matrix $`W \in \mathbb{R}^{d \times d}`$:

$$
\large
\rho(W) = \max_{i} |\lambda_i(W)|
$$

**Symbol Definitions:**
* $`W \in \mathbb{R}^{d \times d}`$: The square weight matrix.
* $`\lambda_i(W) \in \mathbb{C}`$: The $`i`$-th eigenvalue of $`W`$.
* $`\rho(W) \in \mathbb{R}_{\ge 0}`$: The spectral radius (maximum eigenvalue modulus).

**Dynamical Implication:** In un-normalized recurrent loops $`h_t = W h_{t-1} = W^t h_0`$, the state magnitude evolves as $`\|h_t\| \sim \mathcal{O}(\rho(W)^t)`$. If $`\rho(W) \gt 1.0`$, states explode exponentially; if $`\rho(W) \ll 1.0`$, past information is wiped out. `syndset` flags square matrices with $`\rho(W) \gt 1.5`$.

---

### 7. Loss Surface Curvature & Hutchinson's Hessian Trace Estimator

#### The Curvature Formulation
Let $`\mathcal{L}(\theta)`$ be the scalar loss evaluated at parameter vector $`\theta \in \mathbb{R}^P`$. The local curvature is described by the Hessian matrix:

$$
\large
H = \nabla_\theta^2 \mathcal{L}(\theta) \in \mathbb{R}^{P \times P}, \quad [H]_{ij} = \frac{\partial^2 \mathcal{L}}{\partial \theta_i \partial \theta_j}
$$

Because instantiating the full $`P \times P`$ matrix requires $`\mathcal{O}(P^2)`$ memory (petabytes for modern architectures), `syndset` implements **Hutchinson's randomized trace estimator** (Hutchinson, 1989):

$$
\large
\text{Tr}(H) = \mathbb{E}_{v \sim \mathcal{D}} \left[ v^\top H v \right] \approx \frac{1}{M} \sum_{m=1}^M v_m^\top H v_m
$$

where each random vector $`v_m \in \{-1, +1\}^P`$ is sampled from an independent Rademacher distribution.

#### Pearlmutter's Vector-Hessian Product
The matrix-vector product $`H v_m`$ is computed without ever forming $`H`$ using the chain rule identity:

$$
\large
H v_m = \nabla_\theta \left( \nabla_\theta \mathcal{L}(\theta)^\top v_m \right)
$$

**Symbol Definitions:**
* $`P \in \mathbb{N}`$: Total number of trainable parameters in the model.
* $`H \in \mathbb{R}^{P \times P}`$: The Hessian curvature matrix.
* $`\text{Tr}(H) = \sum_{i=1}^P \lambda_i(H)`$: The trace of the Hessian (sum of all curvature eigenvalues).
* $`M \in \mathbb{N}`$: The number of stochastic projection samples (default: 5).
* $`v_m \in \{-1, +1\}^P`$: Random Rademacher probe vector.
* $`\displaystyle \large \bar{\lambda} = \frac{\text{Tr}(H)}{P}`$: The mean curvature eigenvalue.

**Empirical Interpretation:**
* **$`\large \bar{\lambda} \gt 50.0`$ (Hyper-Sharp Minima):** High curvature ravines where gradient descent oscillates violently, sensitive to learning rate and floating point rounding.
* **$`\large \bar{\lambda} \lt -1.0`$ (Negative Curvature):** The loss surface is locally concave, indicating the model is initialized near a saddle point.

---

### 8. Permutation Equivariance & Invariance

In geometric deep learning (Bronstein et al., 2021), architectures processing sets (DeepSets, Set Transformers) or graphs (GNNs) must respect symmetry under coordinate permutations $`\pi \in \mathcal{S}_n`$.

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
* $`X \in \mathbb{R}^{B \times N \times D}`$: Input tensor, where $`N`$ is the number of elements/nodes along permutation dimension `perm_dim`.
* $`\pi \in \mathcal{S}_N`$: A random permutation operator that reorders elements along dimension $`N`$.
* $`f`$: The forward function of the neural network.
* $`\displaystyle \large \delta_{\text{perm}} = \frac{\|f(\pi(X)) - \hat{Y}_{\text{expected}}\|_F}{\|\hat{Y}_{\text{expected}}\|_F + 10^{-7}}`$: The relative Frobenius error.

If $`\large \delta_{\text{perm}} \gt 10^{-3}`$, the model inadvertently relies on element ordering (e.g. via unintended positional embeddings or asymmetric pooling).

---

---

### 10. Axiomatic Attribution & Causal Ground-Truth Mechanics

#### The Problem: Evaluating Interpretability Without Ground Truth
Standard interpretability benchmarks evaluate attribution methods on natural datasets (e.g., ImageNet or OpenWebText) where the true underlying causal data-generating distribution is unknown. Consequently, practitioners cannot distinguish whether a noisy saliency map stems from a defective attribution algorithm or an architecture that legitimately relies on spurious background correlations.

`syndset` resolves this by coupling **Axiomatic Attribution** (Sundararajan et al., 2017) with synthetic benchmark datasets that possess **exact analytical causal support masks** $`M^{\ast} \in \{0, 1\}^D`$.

---

#### Integrated Gradients & Path Completeness

Let $`F: \mathbb{R}^D \to \mathbb{R}`$ represent the scalar output score of a neural network (e.g., the target class logit). Let $`x \in \mathbb{R}^D`$ be the input sample, and let $`x' \in \mathbb{R}^D`$ denote a baseline reference input (defaulting to the zero vector $`\mathbf{0}`$).

The straight-line path trajectory between $`x'`$ and $`x`$ is parameterized by $`\alpha \in [0, 1]`$:

$$
\large
\gamma(\alpha) = x' + \alpha (x - x')
$$

The path-integrated gradient attribution of the $`i`$-th input coordinate is defined as:

$$
\large
\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha
$$

**Symbol Definitions:**
* $`x = (x_1, \dots, x_D) \in \mathbb{R}^D`$: The observed input vector of feature dimension $`D`$.
* $`x' = (x'_1, \dots, x'_D) \in \mathbb{R}^D`$: The baseline reference vector representing the neutral absence of signal (defaulting to $`\mathbf{0}`$).
* $`D \in \mathbb{N}`$: Total feature dimensionality (or flattened coordinate count) of the input.
* $`\alpha \in [0, 1]`$: Dimensionless path scalar parameterizing linear interpolation from baseline $`x'`$ ($`\alpha = 0`$) to sample $`x`$ ($`\alpha = 1`$).
* $`\gamma(\alpha) \in \mathbb{R}^D`$: The straight-line path trajectory curve in $`\mathbb{R}^D`$ connecting $`x'`$ to $`x`$.
* $`F: \mathbb{R}^D \to \mathbb{R}`$: The scalar target model function (e.g. logit score for class $`c`$).
* $`\frac{\partial F}{\partial x_i}`$: The partial derivative (gradient) of output $`F`$ with respect to the $`i`$-th input coordinate.
* $`\text{IG}_i(x) \in \mathbb{R}`$: The continuous path-integrated attribution value allocated to feature coordinate $`i`$.

#### The Completeness Axiom
By the Fundamental Theorem of Calculus for line integrals:

$$
\large
\sum_{i=1}^D \text{IG}_i(x) = \int_0^1 \left( \sum_{i=1}^D \frac{\partial F(\gamma(\alpha))}{\partial \gamma_i(\alpha)} \frac{d \gamma_i(\alpha)}{d\alpha} \right) d\alpha = \int_0^1 \frac{d F(\gamma(\alpha))}{d\alpha} d\alpha = F(x) - F(x')
$$

**Numerical Riemann Approximation:**  
In discrete computation, the continuous path integral is evaluated via $`M`$-step Riemann summation:

$$
\large
\text{IG}_i^{\text{approx}}(x) = (x_i - x'_i) \times \frac{1}{M} \sum_{m=1}^M \frac{\partial F\left(x' + \frac{m}{M}(x - x')\right)}{\partial x_i}
$$

**Path Completeness Error:**  
`syndset` evaluates whether an architecture satisfies the Completeness Axiom via the relative discrepancy:

$$
\large
\mathcal{E}_{\text{comp}}(x) = \frac{\left| \sum_{i=1}^D \text{IG}_i^{\text{approx}}(x) - \big(F(x) - F(x')\big) \right|}{\left|F(x) - F(x')\right| + \epsilon}
$$

**Numerical Approximation Symbol Definitions:**
* $`M \in \mathbb{N}`$: Number of discrete Riemann interpolation steps along the path trajectory (default: $`20`$).
* $`m \in \{1, \dots, M\}`$: Integer step index along the interpolation path.
* $`\text{IG}_i^{\text{approx}}(x) \in \mathbb{R}`$: The discrete Riemann-sum approximation of the path integral for feature $`i`$.
* $`\mathcal{E}_{\text{comp}}(x) \in \mathbb{R}_{\ge 0}`$: Dimensionless relative path completeness error.
* $`\epsilon = 10^{-7}`$: Small numerical stabilization constant preventing division by zero when $`F(x) \approx F(x')`$.

* **Failure Diagnostic:** If $`\mathcal{E}_{\text{comp}} \gt 0.30`$, the model violates conservative path integration. This flags non-differentiable activation cliffs, broken custom CUDA autograd formulas, or non-conservative skip-connections.

---

#### Causal Attribution Signal-to-Noise Ratio (SNR)

In benchmark datasets with known active support (e.g., [`SparseXORDataset`](#sparsexordataset)), coordinates are partitioned into two disjoint subsets using the analytical binary mask $`M^{\ast} \in \{0, 1\}^D`$:
1. **Causal Support Set:** $`S_{\text{causal}} = \{i \in \{1, \dots, D\} \mid M^{\ast}_i = 1\}`$ with cardinality $`K = |S_{\text{causal}}|`$.
2. **Distractor Noise Set:** $`S_{\text{distractor}} = \{j \in \{1, \dots, D\} \mid M^{\ast}_j = 0\}`$ with cardinality $`D - K`$.

`syndset` measures the mean absolute attribution density over each coordinate partition:

$$
\large
\bar{A}_{\text{causal}} = \frac{1}{|S_{\text{causal}}|} \sum_{i \in S_{\text{causal}}} |\text{IG}_i(x)|, \qquad \bar{A}_{\text{distractor}} = \frac{1}{|S_{\text{distractor}}|} \sum_{j \in S_{\text{distractor}}} |\text{IG}_j(x)|
$$

The **Attribution Signal-to-Noise Ratio** is defined on a logarithmic decibel scale:

$$
\large
\text{SNR}_{\text{attr}} = 10 \log_{10}\left( \frac{\bar{A}_{\text{causal}} + \epsilon_{\text{snr}}}{\bar{A}_{\text{distractor}} + \epsilon_{\text{snr}}} \right) \quad [\text{dB}]
$$

**Complementary Causal Metrics:**

* **Causal Mass Ratio ($`\rho_{\text{causal}} \in [0, 1]`$):**

$$
\large
\rho_{\text{causal}} = \frac{\sum_{i \in S_{\text{causal}}} |\text{IG}_i(x)|}{\sum_{k=1}^D |\text{IG}_k(x)| + \epsilon_{\text{snr}}}
$$

* **Top-$`K`$ Precision ($`\text{Prec}@K \in [0, 1]`$):**

$$
\large
\text{Prec}@K = \frac{\left| \text{argtop}_K\left(\\{|\text{IG}_k(x)|\\}_{k=1}^D\right) \cap S_{\text{causal}} \right|}{K}
$$

**Causal Benchmark Symbol Definitions:**
* $`M^{\ast} \in \{0, 1\}^D`$: Analytical ground-truth causal support mask, where $`M^{\ast}_i = 1`$ indicates a causal coordinate and $`M^{\ast}_j = 0`$ indicates a distractor.
* $`S_{\text{causal}} \subset \{1, \dots, D\}`$: Set of indices corresponding to true causal coordinates.
* $`S_{\text{distractor}} \subset \{1, \dots, D\}`$: Complementary set of indices corresponding to uninformative distractor noise coordinates.
* $`K = |S_{\text{causal}}| \in \mathbb{N}`$: Total number of ground-truth causal features.
* $`\bar{A}_{\text{causal}} \in \mathbb{R}_{\ge 0}`$: Mean absolute attribution density over causal coordinates.
* $`\bar{A}_{\text{distractor}} \in \mathbb{R}_{\ge 0}`$: Mean absolute attribution density over distractor noise coordinates.
* $`\text{SNR}_{\text{attr}} \in \mathbb{R}`$: Attribution signal-to-noise ratio in decibels (dB).
* $`\rho_{\text{causal}} \in [0, 1]`$: Fraction of total absolute attribution mass concentrated on the causal coordinates.
* $`\text{argtop}_K`$: Set operator returning the $`K`$ coordinate indices with the largest absolute attribution values.
* $`\text{Prec}@K \in [0, 1]`$: Precision of the top-$`K`$ attributed features with respect to the ground-truth causal set.
* $`\epsilon_{\text{snr}} = 10^{-8}`$: Small numerical constant preventing division by zero and $`\log_{10}(0)`$.

**Empirical Health Regimes:**
* $`\text{SNR}_{\text{attr}} \ge 3.0 \text{ dB}`$ and $`\text{Prec}@K \ge 0.8`$: **Healthy**. The architecture concentrates attention and gradients cleanly on causal coordinates.
* $`0.0 \text{ dB} \le \text{SNR}_{\text{attr}} \lt 3.0 \text{ dB}`$: **Warning**. Weak inductive bias; uninformative noise features corrupt internal representations.
* $`\text{SNR}_{\text{attr}} \lt 0.0 \text{ dB}`$: **Failed**. Pathological inversion; the model places higher attribution on distractor noise than on the actual ground-truth causal signal.

---

#### Layer Conductance & Algorithmic Depth Decomposition

To determine *at which depth* algorithmic decisions are synthesized, `syndset` implements **Layer Conductance** (Dhamdhere et al., 2018).

Let $`h_l \in \mathbb{R}^{B \times D_l}`$ denote the activation tensor of intermediate layer $`l \in \{1, \dots, L\}`$ under forward input $`x`$. The path-integrated conductance tensor of layer $`l`$ is:

$$
\large
\text{Cond}(h_l) = (h_l(x) - h_l(x')) \odot \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial h_l} d\alpha
$$

The scalar conductance mass of layer $`l`$ is its entrywise $`L_1`$ norm:

$$
\large
C(l) = \\|\text{Cond}(h_l)\\|_1 = \sum_{b=1}^B \sum_{d=1}^{D_l} |\text{Cond}(h_l)_{b, d}|
$$

The normalized depth attribution share is:

$$
\large
P(l) = \frac{C(l)}{\sum_{k=1}^L C(k) + \epsilon_{\text{snr}}} \in [0, 1], \qquad \sum_{l=1}^L P(l) = 1
$$

**Layer Conductance Symbol Definitions:**
* $`L \in \mathbb{N}`$: Total number of hooked intermediate layers evaluated in the architecture.
* $`l \in \{1, \dots, L\}`$: Layer index.
* $`B \in \mathbb{N}`$: Batch size of input samples.
* $`D_l \in \mathbb{N}`$: Output feature dimensionality (channel width or hidden units) of layer $`l`$.
* $`h_l(x) \in \mathbb{R}^{B \times D_l}`$: Activation tensor emitted by layer $`l`$ under input sample $`x`$.
* $`h_l(x') \in \mathbb{R}^{B \times D_l}`$: Baseline activation tensor emitted by layer $`l`$ under baseline reference $`x'`$.
* $`\odot`$: Elementwise Hadamard tensor product.
* $`\frac{\partial F}{\partial h_l} \in \mathbb{R}^{B \times D_l}`$: Upstream gradient tensor backpropagated from scalar output $`F`$ to intermediate layer activations $`h_l`$.
* $`\text{Cond}(h_l) \in \mathbb{R}^{B \times D_l}`$: Layer conductance tensor quantifying the directional flow of attribution through layer $`l`$.
* $`C(l) \in \mathbb{R}_{\ge 0}`$: Total scalar conductance mass (sum of absolute conductance entries) for layer $`l`$.
* $`P(l) \in [0, 1]`$: Normalized proportion of total network attribution flowing through layer $`l`$.
* $`\bar{P} = \frac{1}{L}`$: The uniform theoretical fair-share allocation across $`L`$ layers.

**Dead Layer & Effective Depth Diagnostics:**
* **Fair-Share Baseline:** In an idealized uniform network of $`L`$ layers, each layer contributes $`\bar{P} = \frac{1}{L}`$.
* **Dead Layer Condition:** Any layer with conductance share:

$$
\large
P(l) \lt 0.01 \times \frac{1}{L}
$$

is flagged as a **dead / bypassed layer** (contributes less than 1% of expected fair share).

* **Effective Depth Ratio (EDR):**

$$
\large
\text{EDR} = \frac{1}{L} \sum_{l=1}^L \mathbf{1}_{\left\\{ P(l) \ge 0.10 \times \frac{1}{L} \right\\}}
$$

where $`\mathbf{1}_{\{\cdot\}} \in \{0, 1\}`$ is the indicator function. If $`\text{EDR} \lt 0.50`$, more than half the layers in the architecture are redundant passthroughs.

---

#### Context Attribution Decay & Recency Bias

In sequence architectures (Transformers, State-Space Models, RNNs), long-context memory can fail due to **attention sinks** or **exponential recency collapse**.

Let $`x \in \mathbb{Z}^{B \times T}`$ be an input token sequence of length $`T`$ across batch size $`B`$. Token attributions $`\text{Attr} \in \mathbb{R}^{B \times T}`$ are computed across the continuous embedding representations:

$$
\large
A(t) = \frac{1}{B} \sum_{b=1}^B |\text{Attr}_{b, t}|, \qquad t \in \\{0, \dots, T-1\\}
$$

`syndset` computes the **Recency Concentration Ratio**:

$$
\large
R_{\text{recency}} = \frac{\sum_{t = \lfloor 0.8 T \rfloor}^{T-1} A(t)}{\sum_{t=0}^{T-1} A(t) + \epsilon_{\text{snr}}} \in [0, 1]
$$

**Context Attribution Symbol Definitions:**
* $`B \in \mathbb{N}`$: Batch size of sequence samples.
* $`T \in \mathbb{N}`$: Sequence length (total number of tokens per sequence).
* $`t \in \{0, \dots, T-1\}`$: Discrete token temporal index, where $`t = 0`$ represents the earliest token and $`t = T - 1`$ represents the most recent token.
* $`x \in \mathbb{Z}^{B \times T}`$: Input discrete token index tensor.
* $`\text{Attr}_{b, t} \in \mathbb{R}`$: Scalar attribution value allocated to token position $`t`$ for sample $`b`$, aggregated across embedding channels.
* $`A(t) \in \mathbb{R}_{\ge 0}`$: Batch-averaged attribution mass assigned to temporal position $`t`$.
* $`\lfloor \cdot \rfloor`$: Mathematical floor operator, rounding down to the nearest integer.
* $`R_{\text{recency}} \in [0, 1]`$: Fraction of total sequence attribution concentrated in the final 20% of tokens ($`t \ge \lfloor 0.8 T \rfloor`$).

**Failure Criteria:**
* **Early Context Blindness:**

$$
\large
\sum_{t=0}^{\lfloor 0.2 T \rfloor} A(t) \lt 10^{-7}
$$

Flags that tokens in the first 20% of the sequence receive zero gradient attribution (complete loss of early context).
* **Severe Recency Bias:** If $`R_{\text{recency}} \gt 0.95`$, the architecture concentrates over 95% of its attribution on the final 20% of tokens, indicating broken long-range memory retention.

---

### 11. Centered Kernel Alignment (CKA) & Representation Dynamics

#### The Invariance Problem in Representational Similarity
When comparing intermediate representations across layers or distinct architectures, canonical correlation analysis (CCA) is hypersensitive to small numerical perturbations in wide layers ($`D \gt N`$). Standard Euclidean distance fails because representations are invariant to orthogonal rotation and uniform scaling.

#### Hilbert-Schmidt Independence Criterion (HSIC)
Let $`X \in \mathbb{R}^{N \times D_1}`$ and $`Y \in \mathbb{R}^{N \times D_2}`$ denote activation matrices from two layers across $`N`$ samples. Let $`K = X X^\top \in \mathbb{R}^{N \times N}`$ and $`L = Y Y^\top \in \mathbb{R}^{N \times N}`$ be the corresponding linear kernel Gram matrices.

The empirical Hilbert-Schmidt Independence Criterion is:

$$
\large
\text{HSIC}(K, L) = \frac{1}{(N-1)^2} \text{Tr}(K H L H)
$$

where $`H \in \mathbb{R}^{N \times N}`$ is the empirical centering projection matrix:

$$
\large
H = I_N - \frac{1}{N} \mathbf{1}_N \mathbf{1}_N^\top
$$

#### Linear CKA Formulation
`syndset` implements **Linear CKA** (Kornblith et al., ICML 2019):

$$
\large
\text{CKA}(X, Y) = \frac{\text{HSIC}(X X^\top, Y Y^\top)}{\sqrt{\text{HSIC}(X X^\top, X X^\top) \cdot \text{HSIC}(Y Y^\top, Y Y^\top)}}
$$

By substituting centered feature representations $`X_c = H X`$ and $`Y_c = H Y`$:

$$
\large
\text{Tr}(K H L H) = \text{Tr}(X X^\top H Y Y^\top H) = \\|Y_c^\top X_c\\|_F^2
$$

Yielding the explicit sample-covariance quotient:

$$
\large
\text{CKA}(X, Y) = \frac{\\|Y_c^\top X_c\\|_F^2}{\\|X_c^\top X_c\\|_F \cdot \\|Y_c^\top Y_c\\|_F} \in [0, 1]
$$

**CKA Symbol Definitions:**
* $`N \in \mathbb{N}`$: Number of evaluated probe samples ($`N \ge 2`$).
* $`D_1, D_2 \in \mathbb{N}`$: Feature dimensions of layers $`X`$ and $`Y`$.
* $`X_c = X - \frac{1}{N}\mathbf{1}\mathbf{1}^\top X`$: Column-centered sample activation matrix of layer $`X`$.
* $`Y_c = Y - \frac{1}{N}\mathbf{1}\mathbf{1}^\top Y`$: Column-centered sample activation matrix of layer $`Y`$.
* $`\|\cdot\|_F`$: Matrix Frobenius norm, $`\|A\|_F = \sqrt{\sum_{i,j} A_{i,j}^2}`$.
* $`\text{CKA}(X, Y) \in [0, 1]`$: Continuous alignment index, where 1.0 denotes identical representational geometry up to orthogonal rotation and scaling, and 0.0 denotes complete orthogonality.

**Theoretical Invariance Properties:**
* **Orthogonal Invariance:** For any orthogonal matrices $`Q_1 \in O(D_1)`$ and $`Q_2 \in O(D_2)`$, $`\text{CKA}(X Q_1, Y Q_2) = \text{CKA}(X, Y)`$.
* **Isotropic Scaling Invariance:** For any non-zero scalars $`\alpha, \beta \in \mathbb{R} \setminus \{0\}`$, $`\text{CKA}(\alpha X, \beta Y) = \text{CKA}(X, Y)`$.

**Diagnostic Regimes:**
* **Consecutive Layer Redundancy ($`\text{CKA}(l, l+1) \gt 0.99`$):** The representation undergoes negligible transformation. Depth is wasted on identity-like pass-through computation.
* **Representation Discontinuity ($`\text{CKA}(l, l+1) \lt 0.03`$):** Severe information bottleneck; representations diverge abruptly without gradual refinement.

---

### 12. Data-Free Synaptic Flow Conservation (SynFlow)

#### The Problem: Data-Free Structural Path Bottlenecks
Before training data is ingested, an architecture must have topological pathways capable of transmitting signals from the loss back to all parameters without artificial layer starvation or disconnected subgraphs.

#### Synaptic Saliency Formulation
`syndset` implements **SynFlow** (Tanaka et al., NeurIPS 2020). The probe evaluates parameter saliency without data by replacing all parameter tensors with their elementwise absolute values $`\theta' = |\theta|`$ and passing an all-ones input vector $`\mathbf{1}`$:

$$
\large
\mathcal{R}(\theta) = \mathbf{1}^\top f(\mathbf{1}; |\theta|) = \sum_{k=1}^K f_k(\mathbf{1}; |\theta|)
$$

The data-free synaptic saliency tensor for layer parameters $`\theta_l`$ is defined as the entrywise product of weights and upstream gradients:

$$
\large
\mathcal{S}(\theta_l) = \sum_{p} \left| \theta_{l, p} \odot \frac{\partial \mathcal{R}}{\partial \theta_{l, p}} \right|
$$

Total network synaptic flow is the sum across all $`L`$ parameterized layers:

$$
\large
\mathcal{S}_{\text{total}} = \sum_{l=1}^L \mathcal{S}(\theta_l)
$$

**SynFlow Symbol Definitions:**
* $`\theta \in \mathbb{R}^P`$: Total network parameter vector.
* $`|\theta| \in \mathbb{R}_{\ge 0}^P`$: Non-negative elementwise parameter tensor ensuring non-negative activation flow.
* $`\mathbf{1}`$: All-ones input tensor matching model input dimensions.
* $`\mathcal{R} \in \mathbb{R}_{\ge 0}`$: Scalar sum of network outputs under unit input.
* $`\mathcal{S}(\theta_l) \in \mathbb{R}_{\ge 0}`$: Total synaptic flow passing through layer $`l`$.
* $`\mathcal{S}_{\text{total}} \in \mathbb{R}_{\ge 0}`$: Total conserved synaptic flow of the architecture.

**Failure Regimes:**
* **Total Disconnection ($`\mathcal{S}_{\text{total}} \le 10^{-12}`$):** Output is completely independent of parameters or zeroed out by broken computational graphs $`\to`$ **FATAL (VETO)**.
* **Layer Starvation ($`\mathcal{S}(\theta_l) \le 10^{-12}`$ while $`\mathcal{S}_{\text{total}} \gt 0`$):** Specific branch or layer cannot backpropagate gradients $`\to`$ **WARNING / DROP**.

---

### 13. Activation Kurtosis & Low-Precision PTQ Viability

#### The Mechanics of Precision Collapse in INT8 / FP8
In uniform symmetric Post-Training Quantization (PTQ), floating-point tensor values $`X`$ are mapped to $`b`$-bit integer bins $`q \in [-2^{b-1}, 2^{b-1}-1]`$ via scalar step size $`\Delta`$:

$$
\large
\Delta = \frac{2 \max |X|}{2^b - 1}, \qquad Q(X) = \text{clamp}\left(\left\lfloor \frac{X}{\Delta} \right\rceil, -2^{b-1}, 2^{b-1}-1\right)
$$

The expected quantization roundoff error is directly proportional to $`\Delta^2`$:

$$
\large
\mathbb{E}[(X - Q(X) \Delta)^2] \approx \frac{\Delta^2}{12} \propto \frac{(\max |X|)^2}{2^{2b}}
$$

If intermediate layer activations contain extreme outliers ($`\max |X| \gg \sigma_X`$), the quantization grid step $`\Delta`$ expands to cover the outlier, forcing the vast majority of informative features into 0 or 1 discrete bins.

#### Sample Kurtosis Formulation
`syndset` computes the **sample kurtosis** (fourth standardized moment) across intermediate layer activations:

$$
\large
\text{Kurt}(X) = \frac{\frac{1}{M} \sum_{i=1}^M (x_i - \bar{x})^4}{\left( \frac{1}{M} \sum_{i=1}^M (x_i - \bar{x})^2 \right)^2}
$$

and the extreme outlier ratio beyond $`k`$ standard deviations:

$$
\large
\rho_{\text{outlier}}(k) = \frac{1}{M} \sum_{i=1}^M \mathbf{1}_{\\{|x_i - \bar{x}| \gt k \cdot \sigma_X\\}}
$$

**Quantization Symbol Definitions:**
* $`M \in \mathbb{N}`$: Total number of scalar activation elements in a layer tensor across the batch.
* $`\bar{x} = \frac{1}{M} \sum x_i`$: Sample mean of activations.
* $`\sigma_X = \sqrt{\frac{1}{M} \sum (x_i - \bar{x})^2}`$: Sample standard deviation.
* $`\text{Kurt}(X) \in [1, \infty)`$: Sample kurtosis (Gaussian distribution has $`\text{Kurt} = 3.0`$).
* $`\rho_{\text{outlier}}(6) \in [0, 1]`$: Fraction of activation entries exceeding $`6\sigma_X`$.

**Quantization Health Regimes:**
* $`\text{Kurt}(X) \le 15.0`$ and $`\rho_{\text{outlier}}(6) = 0`$: **Healthy**. Direct PTQ to INT8 / FP8 without accuracy loss.
* $`15.0 \lt \text{Kurt}(X) \le 50.0`$: **Warning**. Mild outlier emergence; requires channel-wise scaling or SmoothQuant.
* $`\text{Kurt}(X) \gt 50.0`$: **Failed / Sensitive**. Severe activation spikes; standard uniform quantization will suffer complete precision collapse.

---

### 14. Empirical Neural Tangent Kernel (NTK) Conditioning

#### Continuous Gradient Flow Dynamics
Under continuous-time gradient descent with learning rate $`\eta`$, parameter trajectories obey $`\frac{d\theta}{dt} = -\eta \nabla_\theta \mathcal{L}`$. For Mean Squared Error loss on target outputs $`Y`$, the evolution of model predictions $`u(t) = f(X; \theta(t))`$ satisfies:

$$
\large
\frac{d u(t)}{dt} = -\eta \cdot \Theta(t) \cdot (u(t) - Y)
$$

where $`\Theta(t) \in \mathbb{R}^{N \times N}`$ is the empirical **Neural Tangent Kernel (NTK)** (Jacot et al., NeurIPS 2018):

$$
\large
\Theta_{i, j} = \left\langle \nabla_\theta f(x_i; \theta), \\, \nabla_\theta f(x_j; \theta) \right\rangle = \sum_{p=1}^P \frac{\partial f(x_i)}{\partial \theta_p} \frac{\partial f(x_j)}{\partial \theta_p}
$$

Let $`G \in \mathbb{R}^{N \times P}`$ denote the sample-wise parameter Jacobian matrix $`G_{i, p} = \frac{\partial f(x_i)}{\partial \theta_p}`$. Then $`\Theta = G G^\top`$.

#### Spectral Mode Decomposition & Convergence Disparity
Because $`\Theta`$ is symmetric positive semi-definite, its spectral decomposition is:

$$
\large
\Theta = Q \Lambda Q^\top, \qquad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_N), \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_N \ge 0
$$

Projecting the residual error vector $`e(t) = u(t) - Y`$ onto the orthonormal eigenbasis $`v_k = Q^\top e(t)`$:

$$
\large
v_k(t) = v_k(0) \cdot \exp(-\eta \lambda_k t), \qquad k = 1, \dots, N
$$

The condition number of the empirical NTK governs convergence uniformity:

$$
\large
\kappa(\Theta) = \frac{\lambda_{\max}(\Theta)}{\lambda_{\min}(\Theta) + \epsilon} = \frac{\lambda_1}{\lambda_N + \epsilon}
$$

**NTK Symbol Definitions:**
* $`N \in \mathbb{N}`$: Number of probe samples evaluated.
* $`P \in \mathbb{N}`$: Total number of trainable parameters in the model.
* $`\Theta \in \mathbb{R}^{N \times N}`$: Empirical Neural Tangent Kernel matrix.
* $`\lambda_k \in \mathbb{R}_{\ge 0}`$: $`k`$-th eigenvalue of the NTK, governing the convergence rate of the $`k`$-th error mode.
* $`\kappa(\Theta) \in [1, \infty)`$: NTK condition number.

**Optimization Regimes:**
* $`\kappa(\Theta) \le 10^3`$: **Uniform Fast Convergence**. All error modes converge at comparable exponential rates.
* $`10^3 \lt \kappa(\Theta) \le 10^5`$: **Moderate Stiffness**. Slower high-frequency adaptation.
* $`\kappa(\Theta) \gt 10^5`$: **Severe Ill-Conditioning**. The smallest error components require $`\frac{\lambda_1}{\lambda_N} \gt 10^5`$ times more gradient steps to fit $`\to`$ **DROP**.

---

### 15. Effective Receptive Field (ERF) Spatial Dispersion

#### Gradient-Based Receptive Field Formulation
In vision backbones (CNNs, Vision Transformers), the theoretical receptive field grows linearly with depth, but the **Effective Receptive Field (ERF)** (Luo et al., NeurIPS 2016) follows a 2D Gaussian distribution.

Let $`y_{\text{center}}`$ denote the central scalar activation of the output feature map, and $`x \in \mathbb{R}^{C \times H \times W}`$ the input image. The entrywise gradient intensity is:

$$
\large
G(u, v) = \frac{1}{C} \sum_{c=1}^C \left| \frac{\partial y_{\text{center}}}{\partial x_{c, u, v}} \right|, \qquad u \in \\{0, \dots, H-1\\}, \\; v \in \\{0, \dots, W-1\\}
$$

Normalizing into a 2D probability distribution:

$$
\large
p(u, v) = \frac{G(u, v)}{\sum_{u'=0}^{H-1} \sum_{v'=0}^{W-1} G(u', v') + \epsilon}
$$

The center of mass and spatial variance along both axes are:

$$
\large
\bar{u} = \sum_{u, v} u \cdot p(u, v), \qquad \sigma_u^2 = \sum_{u, v} (u - \bar{u})^2 p(u, v)
$$

$$
\large
\bar{v} = \sum_{u, v} v \cdot p(u, v), \qquad \sigma_v^2 = \sum_{u, v} (v - \bar{v})^2 p(u, v)
$$

`syndset` computes the **Normalized ERF Radius Ratio**:

$$
\large
\rho_{\text{ERF}} = \frac{2 \sqrt{\sigma_u^2 + \sigma_v^2}}{\max(H, W)} \in [0, 1]
$$

and the **High-Frequency Checkerboard Noise Ratio**:

$$
\large
\chi = \frac{\sum_{u} \sum_{v} |G(u+1, v) - G(u, v)| + |G(u, v+1) - G(u, v)|}{2 \sum_{u, v} G(u, v) + \epsilon} \in [0, 1]
$$

**ERF Symbol Definitions:**
* $`H, W \in \mathbb{N}`$: Height and width of the input image in pixels.
* $`G(u, v) \in \mathbb{R}_{\ge 0}`$: Absolute backpropagated gradient intensity at pixel $`(u, v)`$.
* $`\sigma_u, \sigma_v \in \mathbb{R}_{\ge 0}`$: Effective spread standard deviation along vertical and horizontal axes.
* $`\rho_{\text{ERF}} \in [0, 1]`$: Fraction of spatial image dimension covered by the $`2\sigma`$ effective receptive field.
* $`\chi \in [0, 1]`$: Spatial high-frequency alternating variation ratio (checkerboard grid artifact indicator).

**Diagnostic Thresholds:**
* $`\rho_{\text{ERF}} \lt 0.15`$: **Constricted Receptive Field**. Model is spatially near-sighted and cannot model global context.
* $`\chi \gt 0.60`$: **Checkerboard Aliasing Artifacts**. Caused by unaligned strided convolutions or sub-pixel deconvolution grids.

---

### 16. Multi-Criteria Pareto Dominance & Decision Mechanics

#### Multi-Objective Candidate Evaluation
In architecture selection, competing criteria must be evaluated simultaneously. For each candidate $`i \in \{1, \dots, M\}`$, the diagnostic engine constructs an objective vector:

$$
\large
\mathbf{m}_i = \left( \mathcal{T}_i, \\; \mathcal{E}_i, \\; -\mathcal{C}_i, \\; \mathcal{Q}_i \right)
$$

where $`\mathcal{T}_i \in [0, 1]`$ is the trainability score (gradient flow & curvature flatness), $`\mathcal{E}_i \in [0, 1]`$ is the expressivity score (effective rank & overfit rate), $`\mathcal{C}_i \in [0, 1]`$ is normalized parameter cost, and $`\mathcal{Q}_i \in [0, 1]`$ is quantization readiness.

#### Strict Pareto Dominance
Candidate $`B`$ **strictly dominates** candidate $`A`$ ($`B \succ A`$) if and only if:

$$
\large
\forall k \in \\{1, \dots, K\\}, \quad m_{B, k} \ge m_{A, k} \quad \text{and} \quad \exists k \in \\{1, \dots, K\\} \quad \text{s.t.} \quad m_{B, k} \gt m_{A, k}
$$

The non-dominated Pareto frontier is:

$$
\large
\mathcal{P} = \\{ A \in \mathcal{C} \mid \nexists B \in \mathcal{C} \text{ s.t. } B \succ A \\}
$$

Any architecture $`A \notin \mathcal{P}`$ is suboptimal across all trade-offs and is assigned verdict **🔴 DROP**.

#### TOPSIS Composite Utility Formulation
For non-vetoed candidates, `syndset` evaluates utility using distance to the Positive Ideal Solution $`\mathbf{s}^+`$ and Negative Ideal Solution $`\mathbf{s}^-`$ with profile weights $`\mathbf{w}`$:

$$
\large
d_i^+ = \sqrt{\sum_{k=1}^K w_k (m_{i, k} - s_k^+)^2}, \qquad d_i^- = \sqrt{\sum_{k=1}^K w_k (m_{i, k} - s_k^-)^2}
$$

The composite utility index is:

$$
\large
U_i = \frac{d_i^-}{d_i^+ + d_i^-} \in [0, 1]
$$

Architectures with clean health in $`\mathcal{P}`$ and $`U_i \ge 0.80`$ receive the verdict **🟢 PROMOTE**.

---
