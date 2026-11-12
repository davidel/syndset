"""Diagnostics for evaluating layer representational similarity using CKA."""

import torch
import torch.nn as nn

from syndset.diagnostics.representation import _flatten_activation_to_2d
from syndset.utils.hooks import ForwardActivationHook


def compute_linear_cka(features_x, features_y, eps=1e-10):
  """Computes linear Centered Kernel Alignment (CKA) between two feature matrices.

  Theoretical background:
    Linear CKA (Kornblith et al., ICML 2019) measures representational similarity
    invariant to orthogonal transformation and isotropic scaling:
      CKA(X, Y) = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c and Y_c are column-centered sample-by-feature activation matrices.

  Args:
    features_x: A 2D torch.Tensor of shape (N, D1).
    features_y: A 2D torch.Tensor of shape (N, D2).
    eps: Small float to prevent division by zero.

  Returns:
    A float between 0.0 and 1.0 indicating representational similarity.
  """
  if features_x.dim() != 2 or features_y.dim() != 2:
    return 0.0

  if features_x.shape[0] != features_y.shape[0] or features_x.shape[0] < 2:
    return 0.0

  x = features_x.float()
  y = features_y.float()

  # Column-center the activations: X_c = X - mean(X)
  x_c = x - x.mean(dim=0, keepdim=True)
  y_c = y - y.mean(dim=0, keepdim=True)

  # HSIC(K, L) for linear kernel: ||Y_c^T X_c||_F^2
  cross_term = torch.norm(torch.matmul(y_c.t(), x_c), p="fro")**2
  denom_x = torch.norm(torch.matmul(x_c.t(), x_c), p="fro")
  denom_y = torch.norm(torch.matmul(y_c.t(), y_c), p="fro")

  denominator = (denom_x * denom_y).item()
  if denominator <= eps:
    return 0.0

  similarity = (cross_term.item() / denominator)
  return float(max(0.0, min(1.0, similarity)))


def check_layer_cka(
  model,
  sample_input,
  target_types=None,
  exclude_types=(nn.Embedding, nn.EmbeddingBag),
  min_dissimilarity=0.03,
  max_redundancy=0.99,
):
  """Evaluates pairwise layer similarity and identifies wasted depth or representational gaps.

  Collects forward activations across targeted submodules (filtering by nn.Module
  subclasses or excluding input lookups like embeddings) and computes the full
  layer-by-layer CKA similarity matrix. Flags consecutive layers with:
    - Redundant representations: CKA > max_redundancy (e.g. >0.99)
    - Disconnected representations: CKA < min_dissimilarity (e.g. <0.03)

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: A torch.Tensor or tuple/dict matching model forward inputs.
    target_types: Optional module types or subclasses to inspect
      (e.g. (nn.Linear, nn.TransformerEncoderLayer)). If None, inspects parameterized
      layers that are not in `exclude_types`.
    exclude_types: Module subclasses to exclude from comparison
      (default: (nn.Embedding, nn.EmbeddingBag)). Prevents non-sequential input lookup
      layers from being falsely flagged as disconnected.
    min_dissimilarity: Minimum acceptable CKA between consecutive layers.
    max_redundancy: Threshold above which consecutive layers are deemed redundant.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'layer_names': List of hooked layer names.
      - 'cka_matrix': 2D list of pairwise CKA values between layers.
      - 'consecutive_similarities': Dict mapping (layer_i, layer_{i+1}) to CKA float.
      - 'redundant_layers': List of consecutive layer pairs with excessive similarity.
      - 'disconnected_layers': List of consecutive layer pairs with abrupt divergence.
      - 'issues': Actionable diagnostic recommendations.
  """
  model.eval()

  with ForwardActivationHook(model, target_types=target_types, exclude_types=exclude_types) as hook:
    with torch.no_grad():
      if isinstance(sample_input, dict):
        _ = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        _ = model(*sample_input)
      else:
        _ = model(sample_input)

    activations = hook.activations

  layer_names = list(activations.keys())
  num_layers = len(layer_names)

  if num_layers < 2:
    return {
      "status": "healthy",
      "layer_names": layer_names,
      "cka_matrix": [[1.0]] if num_layers == 1 else [],
      "consecutive_similarities": {},
      "redundant_layers": [],
      "disconnected_layers": [],
      "issues": ["Fewer than 2 parameterized layers found; CKA pairwise analysis skipped."],
    }

  # Flatten all activations to 2D
  flat_acts = {}
  for name, act in activations.items():
    flat_acts[name] = _flatten_activation_to_2d(act)

  # Compute CKA matrix
  cka_matrix = [[0.0] * num_layers for _ in range(num_layers)]
  for i in range(num_layers):
    cka_matrix[i][i] = 1.0
    for j in range(i + 1, num_layers):
      val = compute_linear_cka(flat_acts[layer_names[i]], flat_acts[layer_names[j]])
      cka_matrix[i][j] = val
      cka_matrix[j][i] = val

  consecutive_similarities = {}
  redundant_layers = []
  disconnected_layers = []
  issues = []

  for i in range(num_layers - 1):
    name_a = layer_names[i]
    name_b = layer_names[i + 1]
    sim = cka_matrix[i][i + 1]
    pair_key = f"{name_a} -> {name_b}"
    consecutive_similarities[pair_key] = sim

    if sim > max_redundancy:
      redundant_layers.append(pair_key)
      issues.append(f"Layers '{pair_key}' exhibit near-identical representations "
                    f"(CKA={sim:.3f} > {max_redundancy}). Depth may be redundant.")
    elif sim < min_dissimilarity:
      disconnected_layers.append(pair_key)
      issues.append(f"Layers '{pair_key}' suffer sudden representational disconnect "
                    f"(CKA={sim:.3f} < {min_dissimilarity}).")

  if disconnected_layers:
    status = "failed"
  elif redundant_layers:
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "layer_names": layer_names,
    "cka_matrix": cka_matrix,
    "consecutive_similarities": consecutive_similarities,
    "redundant_layers": redundant_layers,
    "disconnected_layers": disconnected_layers,
    "issues": issues,
  }
