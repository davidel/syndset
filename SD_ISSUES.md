# Synthetic Diagnostics (SD) Issues & Fix Tracker

This document records the resolved diagnostic and regression issues verified by `tests/test_sd_issues_fixes.py`.

## Resolved Issues

- **A1: Causal LM 3D Tensor Output Handling**
  - *Problem*: Models outputting 3D logits `(batch, seq_len, vocab_size)` crashed when evaluated against 1D sequence targets with `RuntimeError: Expected target size [8, 256], got [8]`.
  - *Fix*: Added automatic normalization to extract final token predictions `pred[:, -1, :]` in diagnostics and loss computation.

- **A2: SDPA Curvature Derivative Failure on CPU**
  - *Problem*: PyTorch CPU attention kernels lacked second-order derivatives (`create_graph=True`), raising exceptions during Hessian-vector calculations.
  - *Fix*: Implemented `_get_sdpa_context` selecting `SDPBackend.MATH` on CPU.

- **B1: CKA False Disconnections on Embeddings**
  - *Problem*: CKA flagged embedding layers with consecutive zero similarities as structurally disconnected.
  - *Fix*: Excluded embedding layers by default from intermediate representation similarity checks.

- **B2: SynFlow Starvation on Zero Biases**
  - *Problem*: Zero-initialized biases caused false positive layer starvation warnings.
  - *Fix*: Safeguarded SynFlow metric computation to handle zero-initialized biases properly.

- **C1: Tournament Report Sub-Score Visibility**
  - *Problem*: Tournament scorecard tables did not display individual sub-scores (`Train`, `Expr`, `Eff`, `Quant`).
  - *Fix*: Updated `TournamentReport` to render multi-criteria sub-scores.

- **C2: Continuous Overfit Capacity Score**
  - *Problem*: Capacity scoring was a binary cliff rather than reflecting graded loss reduction.
  - *Fix*: Formulated continuous capacity scoring based on relative loss reduction.

- **D1: Gradient SNR Signature Flexibility**
  - *Problem*: `check_gradient_snr` required specific batch structures.
  - *Fix*: Supported both iterable batch collections and `(inputs, targets)` tensor pairs.

- **D2: NTK Conditioning Signature Uniformity**
  - *Problem*: Calling `check_ntk_conditioning` with `sample_target` collided with positional parameters.
  - *Fix*: Standardized signature across diagnostic probes.
