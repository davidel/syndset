"""Representation diagnostics: effective rank and dead unit detection."""

import torch

from syndset.utils.hooks import ForwardActivationHook
from syndset.utils.stats import compute_effective_rank


def _flatten_activation_to_2d(activation):
  """Reshapes layer activations into a 2D sample-by-feature matrix.

  Args:
    activation: A torch.Tensor with 2, 3, 4, or more dimensions.

  Returns:
    A 2D torch.Tensor of shape (num_samples, feature_dim).
  """
  if activation.dim() == 2:
    return activation
  if activation.dim() == 3:
    # Typical sequence tensor: (batch, seq_len, hidden_dim) -> (batch * seq_len, hidden_dim)
    batch_size, seq_len, hidden_dim = activation.shape
    return activation.reshape(batch_size * seq_len, hidden_dim)
  if activation.dim() == 4:
    # Typical image tensor: (batch, channels, H, W) -> average pool -> (batch, channels)
    return activation.mean(dim=(2, 3))
  # Fallback: flatten all trailing dimensions into one feature vector
  return activation.reshape(activation.shape[0], -1)


def check_effective_rank(model, sample_input, target_types=None, min_rank_ratio=0.05):
  """Computes the effective rank of hidden layer representations across a batch.

  Dimensional collapse occurs when deep networks project representations into
  a tiny subspace (e.g. All tokens or samples collapse onto a line). This probe
  computes the spectral entropy-based effective rank of each layer's activation matrix.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: A torch.Tensor or tuple/dict of tensors matching model input.
    target_types: Optional iterable of layer types to monitor (e.g. (nn.Linear, nn.Conv2d)).
      If None, inspects all leaf layers with parameters.
    min_rank_ratio: Threshold below which a layer is flagged for dimensional collapse
      (default: 0.05, or 5% of maximum possible rank).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'layer_ranks': Dict mapping layer name to effective rank float.
      - 'rank_ratios': Dict mapping layer name to effective rank / max rank ratio.
      - 'collapsed_layers': List of layer names exhibiting dimensional collapse.
      - 'issues': Human-readable notes and recommendations.
  """
  model.eval()

  with ForwardActivationHook(model, target_types=target_types) as hook:
    with torch.no_grad():
      if isinstance(sample_input, dict):
        _ = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        _ = model(*sample_input)
      else:
        _ = model(sample_input)

  layer_ranks = {}
  rank_ratios = {}
  collapsed_layers = []
  issues = []

  for name, act in hook.activations.items():
    matrix_2d = _flatten_activation_to_2d(act)
    if matrix_2d.dim() != 2 or matrix_2d.shape[0] < 2 or matrix_2d.shape[1] < 2:
      continue

    erank = compute_effective_rank(matrix_2d)
    max_possible_rank = min(matrix_2d.shape[0], matrix_2d.shape[1])
    ratio = erank / max(max_possible_rank, 1)

    layer_ranks[name] = round(erank, 2)
    rank_ratios[name] = round(ratio, 4)

    if ratio < min_rank_ratio:
      collapsed_layers.append(name)

  if collapsed_layers:
    issues.append(
      f"Dimensional collapse detected in {len(collapsed_layers)} layers "
      f"({', '.join(collapsed_layers[:3])}). Activations are constrained to a low-dimensional "
      "subspace. Consider adding normalization (LayerNorm/RMSNorm) or residual connections."
    )
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "layer_ranks": layer_ranks,
    "rank_ratios": rank_ratios,
    "collapsed_layers": collapsed_layers,
    "issues": issues,
  }


def check_dead_units(
  model, sample_input, target_types=None, variance_eps=1e-8, dead_ratio_threshold=0.3
):
  """Detects inactive or dead units (neurons or channels) across a batch.

  A unit is considered dead if its output values exhibit near-zero variance
  across all batch items, such as in the 'dying ReLU' phenomenon.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: A torch.Tensor or tuple/dict of tensors.
    target_types: Optional iterable of layer types to monitor.
    variance_eps: Threshold variance below which a unit is considered dead.
    dead_ratio_threshold: Fraction of dead units in a layer that triggers a warning.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'layer_dead_fractions': Dict mapping layer name to dead unit ratio (0.0 to 1.0).
      - 'problematic_layers': List of layer names with high dead unit fractions.
      - 'issues': Human-readable explanations.
  """
  model.eval()

  with ForwardActivationHook(model, target_types=target_types) as hook:
    with torch.no_grad():
      if isinstance(sample_input, dict):
        _ = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        _ = model(*sample_input)
      else:
        _ = model(sample_input)

  layer_dead_fractions = {}
  problematic_layers = []
  issues = []

  for name, act in hook.activations.items():
    matrix_2d = _flatten_activation_to_2d(act)
    if matrix_2d.dim() != 2 or matrix_2d.shape[0] < 2:
      continue

    # Variance along the batch dimension (dim 0)
    variances = torch.var(matrix_2d.float(), dim=0)
    dead_count = (variances < variance_eps).sum().item()
    total_units = matrix_2d.shape[1]
    dead_fraction = dead_count / max(total_units, 1)

    layer_dead_fractions[name] = round(dead_fraction, 4)

    if dead_fraction >= dead_ratio_threshold:
      problematic_layers.append(name)

  if problematic_layers:
    issues.append(
      f"High dead units in {len(problematic_layers)} layers "
      f"({', '.join(problematic_layers[:3])}). Check activations or init scale."
    )
    status = "warning"
  else:
    status = "healthy"

  return {
    "status": status,
    "layer_dead_fractions": layer_dead_fractions,
    "problematic_layers": problematic_layers,
    "issues": issues,
  }
