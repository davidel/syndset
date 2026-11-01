"""Diagnostics for feature attribution, layer conductance, and causal faithfulness."""

import math

import torch
import torch.nn as nn


def _extract_primary_tensor(output):
  """Extracts the primary floating-point tensor from model return structures.

  Args:
    output: Model output, which could be a Tensor, tuple/list, or object with .logits.

  Returns:
    A torch.Tensor if found, or None.
  """
  if isinstance(output, torch.Tensor):
    return output
  if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
    return output[0]
  if hasattr(output, "logits"):
    return output.logits
  return None


def _find_embedding_module(model):
  """Finds the first nn.Embedding submodule in the model, if present.

  Args:
    model: The torch.nn.Module to inspect.

  Returns:
    The first nn.Embedding submodule found, or None.
  """
  for module in model.modules():
    if isinstance(module, nn.Embedding):
      return module
  return None


def compute_integrated_gradients(
  model,
  sample_input,
  target=None,
  baseline=None,
  num_steps=20,
  embedding_module=None,
):
  """Computes path-integrated gradients along a linear trajectory from baseline to input.

  Theoretical background:
    Axiomatic attribution (Sundararajan et al., 2017) defines the attribution of feature i
    along the straight-line path from baseline x' to input x as:
      IG_i(x) = (x_i - x'_i) * int_0^1 (d F(x' + alpha * (x - x')) / d x_i) d alpha
    which satisfies the Completeness Axiom: sum_i IG_i(x) = F(x) - F(x').

    For discrete token inputs (e.g. integer tokens in LLMs), this function applies
    path integration over the continuous embedding representations.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: Input tensor matching model forward signature.
    target: Optional target tensor or class indices. If provided and 1D integer,
      gathers the corresponding output logits; otherwise sums all outputs.
    baseline: Baseline tensor with same shape as sample_input. Defaults to zeros.
    num_steps: Number of Riemann sum interpolation steps (default: 20).
    embedding_module: Optional nn.Embedding layer to hook for discrete token inputs.

  Returns:
    A tuple of (attributions, completeness_error) where:
      - attributions: Tensor matching sample_input shape (or embedding dimension).
      - completeness_error: Relative discrepancy between sum(attributions) and delta_F.
  """
  model.eval()

  is_discrete = isinstance(sample_input, torch.Tensor) and sample_input.dtype in (
    torch.long,
    torch.int32,
    torch.int64,
  )

  if is_discrete:
    target_emb = embedding_module or _find_embedding_module(model)
    if target_emb is None:
      raise ValueError(
        "sample_input contains integer tokens but no nn.Embedding module was found or provided."
      )

    emb_holder = {}

    def emb_forward_hook(module, inps, output):
      emb_holder["emb"] = output

    handle = target_emb.register_forward_hook(emb_forward_hook)

    # Capture actual embedding outputs
    with torch.no_grad():
      if isinstance(sample_input, dict):
        base_out = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        base_out = model(*sample_input)
      else:
        base_out = model(sample_input)
      actual_emb = emb_holder["emb"].clone()

    handle.remove()

    if baseline is None:
      base_emb = torch.zeros_like(actual_emb)
    elif isinstance(baseline, torch.Tensor) and baseline.shape == actual_emb.shape:
      base_emb = baseline
    else:
      base_emb = torch.zeros_like(actual_emb)

    delta = actual_emb - base_emb
    accum_grad = torch.zeros_like(actual_emb)
    alphas = torch.linspace(0.0, 1.0, num_steps, device=actual_emb.device)

    # Use forward hook to inject interpolated embedding tensors during trajectory
    current_interp = [None]

    def inject_hook(module, inps, output):
      return current_interp[0]

    inject_handle = target_emb.register_forward_hook(inject_hook)

    try:
      for alpha in alphas:
        interp = (base_emb + alpha * delta).detach().requires_grad_(True)
        current_interp[0] = interp

        model.zero_grad(set_to_none=True)
        if isinstance(sample_input, dict):
          out = model(**sample_input)
        elif isinstance(sample_input, (tuple, list)):
          out = model(*sample_input)
        else:
          out = model(sample_input)

        out_tensor = _extract_primary_tensor(out)
        if target is not None and target.dim() == 1 and out_tensor.dim() >= 2:
          score = out_tensor.gather(1, target.unsqueeze(1)).sum()
        else:
          score = out_tensor.sum()

        (grad,) = torch.autograd.grad(score, interp)
        accum_grad += grad.detach()
    finally:
      inject_handle.remove()

    avg_grad = accum_grad / num_steps
    attributions = (delta * avg_grad).sum(dim=-1)

    # Measure completeness error
    with torch.no_grad():
      out_tensor_actual = _extract_primary_tensor(base_out)
      if target is not None and target.dim() == 1 and out_tensor_actual.dim() >= 2:
        val_actual = out_tensor_actual.gather(1, target.unsqueeze(1)).sum().item()
      else:
        val_actual = out_tensor_actual.sum().item()

      # Baseline evaluation
      current_interp[0] = base_emb
      h_temp = target_emb.register_forward_hook(inject_hook)
      if isinstance(sample_input, dict):
        out_base = model(**sample_input)
      elif isinstance(sample_input, (tuple, list)):
        out_base = model(*sample_input)
      else:
        out_base = model(sample_input)
      h_temp.remove()

      out_tensor_base = _extract_primary_tensor(out_base)
      if target is not None and target.dim() == 1 and out_tensor_base.dim() >= 2:
        val_base = out_tensor_base.gather(1, target.unsqueeze(1)).sum().item()
      else:
        val_base = out_tensor_base.sum().item()

      delta_val = val_actual - val_base
      attr_sum = attributions.sum().item()
      completeness_error = abs(attr_sum - delta_val) / (abs(delta_val) + 1e-7)

    return attributions, completeness_error

  # Continuous float inputs
  inputs = sample_input
  if baseline is None:
    baseline = torch.zeros_like(inputs)

  delta = inputs - baseline
  accum_grad = torch.zeros_like(inputs)
  alphas = torch.linspace(0.0, 1.0, num_steps, device=inputs.device)

  for alpha in alphas:
    interpolated = (baseline + alpha * delta).detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out = model(interpolated)
    out_tensor = _extract_primary_tensor(out)

    if target is not None and target.dim() == 1 and out_tensor.dim() >= 2:
      score = out_tensor.gather(1, target.unsqueeze(1)).sum()
    else:
      score = out_tensor.sum()

    (grad,) = torch.autograd.grad(score, interpolated)
    accum_grad += grad.detach()

  avg_grad = accum_grad / num_steps
  attributions = delta * avg_grad

  with torch.no_grad():
    out_actual = _extract_primary_tensor(model(inputs))
    out_base = _extract_primary_tensor(model(baseline))

    if target is not None and target.dim() == 1 and out_actual.dim() >= 2:
      val_actual = out_actual.gather(1, target.unsqueeze(1)).sum().item()
      val_base = out_base.gather(1, target.unsqueeze(1)).sum().item()
    else:
      val_actual = out_actual.sum().item()
      val_base = out_base.sum().item()

    delta_val = val_actual - val_base
    attr_sum = attributions.sum().item()
    completeness_error = abs(attr_sum - delta_val) / (abs(delta_val) + 1e-7)

  return attributions, completeness_error


def check_attribution_faithfulness(
  model,
  dataset,
  sample_indices=None,
  num_steps=20,
  snr_threshold_db=3.0,
  embedding_module=None,
):
  """Evaluates whether model attributions concentrate on ground-truth causal coordinates.

  Theoretical background:
    In benchmark synthetic datasets (SparseXOR, AssociativeRecall, SelectiveCopy),
    the causal support mask M^* in {0, 1}^D is analytically known.
    An architecture with faithful feature routing should place high attribution
    mass on causal coordinates while suppressing distractor noise.

    This probe computes:
      1. Attribution Signal-to-Noise Ratio (SNR):
         SNR_dB = 10 * log10((mean(|Attr_causal|) + eps) / (mean(|Attr_distractor|) + eps))
      2. Causal Mass Ratio:
         sum(|Attr_causal|) / sum(|Attr_all|)
      3. Top-K Precision:
         Fraction of top-K largest attribution features belonging to true causal coordinates.
      4. Path Completeness Error:
         Verification that Integrated Gradients satisfies the Completeness Axiom.

  Args:
    model: The torch.nn.Module to evaluate.
    dataset: A SyntheticDataset instance implementing causal_mask(idx).
    sample_indices: Optional list of sample indices. Defaults to first 8 samples.
    num_steps: Number of interpolation steps along the baseline path (default: 20).
    snr_threshold_db: Minimum SNR in dB for 'healthy' status (default: 3.0 dB).
    embedding_module: Optional nn.Embedding layer for discrete sequence models.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'attribution_snr_db': Mean attribution SNR in decibels across tested samples.
      - 'causal_mass_ratio': Mean fraction of attribution allocated to causal features.
      - 'topk_precision': Mean precision of top-K attributed coordinates.
      - 'mean_completeness_error': Mean path integration completeness error.
      - 'issues': Human-readable notes and warnings.
  """
  if not hasattr(dataset, "causal_mask"):
    raise ValueError(
      f"Dataset {dataset.__class__.__name__} does not implement causal_mask(idx)."
    )

  if sample_indices is None:
    sample_indices = list(range(min(8, len(dataset))))

  snr_list = []
  mass_ratio_list = []
  topk_list = []
  comp_errors = []
  issues = []

  for idx in sample_indices:
    sample_item = dataset[idx]
    if isinstance(sample_item, (tuple, list)):
      sample_input = sample_item[0]
      sample_target = sample_item[1]
    else:
      sample_input = sample_item
      sample_target = None

    # Ensure batch dimension
    if sample_input.dim() == 1:
      batched_input = sample_input.unsqueeze(0)
    elif sample_input.dim() == 3:  # Image (C, H, W)
      batched_input = sample_input.unsqueeze(0)
    else:
      batched_input = sample_input

    batched_target = sample_target.unsqueeze(0) if sample_target is not None else None

    mask = dataset.causal_mask(idx)
    if mask.dim() < batched_input.dim():
      # Reshape mask to match sample shape
      mask_flat = mask.flatten()
    else:
      mask_flat = mask.flatten()

    attr, comp_err = compute_integrated_gradients(
      model,
      batched_input,
      target=batched_target,
      num_steps=num_steps,
      embedding_module=embedding_module,
    )
    comp_errors.append(comp_err)

    attr_flat = attr.squeeze(0).abs().flatten()
    if attr_flat.shape != mask_flat.shape:
      # If image with multiple channels, aggregate across channels
      if attr.dim() == 4:
        attr_flat = attr.squeeze(0).abs().mean(dim=0).flatten()

    causal_vals = attr_flat[mask_flat]
    distractor_vals = attr_flat[~mask_flat]

    mean_causal = causal_vals.mean().item() if causal_vals.numel() > 0 else 0.0
    mean_distractor = distractor_vals.mean().item() if distractor_vals.numel() > 0 else 1e-8

    snr_db = 10.0 * math.log10((mean_causal + 1e-8) / (mean_distractor + 1e-8))
    snr_list.append(snr_db)

    total_mass = attr_flat.sum().item() + 1e-8
    causal_mass = causal_vals.sum().item()
    mass_ratio_list.append(causal_mass / total_mass)

    k = max(1, causal_vals.numel())
    if attr_flat.numel() >= k:
      topk_indices = set(torch.topk(attr_flat, k=k).indices.tolist())
      causal_indices = set(mask_flat.nonzero(as_tuple=True)[0].tolist())
      precision = len(topk_indices.intersection(causal_indices)) / float(k)
      topk_list.append(precision)

  mean_snr = sum(snr_list) / len(snr_list) if snr_list else 0.0
  mean_mass = sum(mass_ratio_list) / len(mass_ratio_list) if mass_ratio_list else 0.0
  mean_topk = sum(topk_list) / len(topk_list) if topk_list else 0.0
  mean_comp = sum(comp_errors) / len(comp_errors) if comp_errors else 0.0

  status = "healthy"
  if mean_comp > 0.3:
    status = "warning"
    issues.append(
      f"High path completeness error ({mean_comp:.1%}). Model may have non-smooth activations "
      "or broken autograd gradients."
    )

  if mean_snr < 0.0:
    status = "failed"
    issues.append(
      f"Negative attribution SNR ({mean_snr:.2f} dB): model places higher attribution on "
      "distractor noise than on ground-truth causal coordinates."
    )
  elif mean_snr < snr_threshold_db:
    if status != "failed":
      status = "warning"
    issues.append(
      f"Low attribution SNR ({mean_snr:.2f} dB < {snr_threshold_db:.1f} dB threshold). "
      "Feature routing is weak or noisy."
    )

  return {
    "status": status,
    "attribution_snr_db": mean_snr,
    "causal_mass_ratio": mean_mass,
    "topk_precision": mean_topk,
    "mean_completeness_error": mean_comp,
    "issues": issues,
  }


def check_layer_conductance(
  model,
  sample_input,
  target=None,
  target_types=None,
  num_steps=15,
):
  """Evaluates the algorithmic depth profile and layer utilization of a network.

  Theoretical background:
    Layer Conductance (Dhamdhere et al., 2018) decomposes the prediction into intermediate
    layer representations h_l:
      Cond(h_l) = (h_l(x) - h_l(x')) * int_0^1 (d F(x' + alpha * (x - x')) / d h_l) d alpha
    which quantifies how much information flowing through layer l contributes to the output.

    If an architecture suffers from dead residual branches, layer collapse, or premature
    decision bottlenecks, intermediate layers will exhibit near-zero conductance (< 1% of mean).

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: Sample input tensor matching model forward signature.
    target: Optional target tensor or class indices.
    target_types: Optional tuple of layer classes to monitor (e.g. (nn.Linear, nn.Conv2d)).
    num_steps: Number of interpolation steps (default: 15).

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'layer_conductances': Dict mapping layer name to normalized percentage contribution.
      - 'effective_depth_ratio': Fraction of layers contributing at least 10% of expected share.
      - 'has_dead_layers': True if any hooked layer contributes < 1% of expected share.
      - 'issues': Human-readable notes and warnings.
  """
  model.eval()

  inputs = sample_input
  baseline = torch.zeros_like(inputs)
  delta = inputs - baseline

  # 1. Identify hookable modules
  hookable_modules = []
  for name, module in model.named_modules():
    if module == model:
      continue
    if target_types is not None:
      if isinstance(module, tuple(target_types)):
        hookable_modules.append((name, module))
    else:
      has_params = any(p.requires_grad for p in module.parameters(recurse=False))
      has_children = len(list(module.children())) > 0
      if has_params or not has_children:
        hookable_modules.append((name, module))

  if not hookable_modules:
    return {
      "status": "warning",
      "layer_conductances": {},
      "effective_depth_ratio": 1.0,
      "has_dead_layers": False,
      "issues": ["No hookable submodules found for layer conductance evaluation."],
    }

  # 2. Capture baseline activations
  baseline_acts = {}
  active_acts = {}

  def make_rec_hook(holder, name):
    def h(mod, inp, out):
      if isinstance(out, torch.Tensor):
        holder[name] = out.detach()
      elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        holder[name] = out[0].detach()

    return h

  handles = [m.register_forward_hook(make_rec_hook(baseline_acts, n)) for n, m in hookable_modules]
  with torch.no_grad():
    _ = model(baseline)
  for h in handles:
    h.remove()

  handles = [m.register_forward_hook(make_rec_hook(active_acts, n)) for n, m in hookable_modules]
  with torch.no_grad():
    _ = model(inputs)
  for h in handles:
    h.remove()

  # 3. Path integration of gradients w.r.t intermediate activations
  alphas = torch.linspace(0.0, 1.0, num_steps, device=inputs.device)
  layer_accum_grads = {name: torch.zeros_like(active_acts[name]) for name, _ in hookable_modules}

  interp_acts = {}

  def make_interp_hook(name):
    def h(mod, inp, out):
      if isinstance(out, torch.Tensor):
        out.retain_grad()
        interp_acts[name] = out
      elif isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        out[0].retain_grad()
        interp_acts[name] = out[0]

    return h

  handles = [m.register_forward_hook(make_interp_hook(n)) for n, m in hookable_modules]

  try:
    for alpha in alphas:
      interp_acts.clear()
      interpolated = (baseline + alpha * delta).detach().requires_grad_(True)
      model.zero_grad(set_to_none=True)

      out = model(interpolated)
      out_tensor = _extract_primary_tensor(out)

      if target is not None and target.dim() == 1 and out_tensor.dim() >= 2:
        score = out_tensor.gather(1, target.unsqueeze(1)).sum()
      else:
        score = out_tensor.sum()

      # Compute gradients with respect to all layer activations simultaneously
      tensors_to_grad = [interp_acts[n] for n, _ in hookable_modules if n in interp_acts]
      if tensors_to_grad:
        grads = torch.autograd.grad(score, tensors_to_grad, retain_graph=False, allow_unused=True)
        for (n, _), g in zip(hookable_modules, grads):
          if g is not None:
            layer_accum_grads[n] += g.detach()
  finally:
    for h in handles:
      h.remove()

  # 4. Compute conductance magnitudes per layer
  raw_conductance = {}
  for name, _ in hookable_modules:
    act_delta = active_acts[name] - baseline_acts[name]
    avg_grad = layer_accum_grads[name] / num_steps
    cond_tensor = act_delta * avg_grad
    raw_conductance[name] = cond_tensor.abs().sum().item()

  total_cond = sum(raw_conductance.values()) + 1e-8
  norm_conductance = {name: val / total_cond for name, val in raw_conductance.items()}

  # Detect dead layers (< 1% of fair share 1 / N)
  expected_share = 1.0 / len(norm_conductance)
  dead_threshold = 0.01 * expected_share
  active_threshold = 0.10 * expected_share

  dead_layers = [name for name, share in norm_conductance.items() if share < dead_threshold]
  active_layers = [name for name, share in norm_conductance.items() if share >= active_threshold]

  effective_depth_ratio = len(active_layers) / len(norm_conductance)

  issues = []
  status = "healthy"
  if dead_layers:
    status = "warning"
    issues.append(
      f"Detected {len(dead_layers)} underutilized/dead layer(s) with < 1% expected conductance: "
      f"{dead_layers[:3]}"
    )

  if effective_depth_ratio < 0.5:
    status = "failed"
    issues.append(
      f"Low effective depth ratio ({effective_depth_ratio:.1%}): more than half of the layers "
      "are bypassed or contribute negligibly to final output."
    )

  return {
    "status": status,
    "layer_conductances": norm_conductance,
    "effective_depth_ratio": effective_depth_ratio,
    "has_dead_layers": len(dead_layers) > 0,
    "issues": issues,
  }


def check_context_attribution_decay(
  model,
  sample_input,
  target=None,
  num_steps=10,
  embedding_module=None,
):
  """Evaluates sequence token attribution profile across temporal/context positions.

  Theoretical background:
    In sequence models (Transformers, SSMs, RNNs), an architecture intended for
    long-context memory can suffer from recency bias (attention sinks, exponential decay).
    This probe computes the integrated gradient attribution profile across sequence
    indices t in {0, ..., T-1} and detects whether the architecture is blind to early context.

  Args:
    model: The torch.nn.Module to evaluate.
    sample_input: 2D integer or 3D float sequence tensor of shape (batch, seq_len, ...).
    target: Optional target tensor.
    num_steps: Number of interpolation steps (default: 10).
    embedding_module: Optional nn.Embedding layer for integer token inputs.

  Returns:
    A dictionary containing:
      - 'status': One of 'healthy', 'warning', or 'failed'.
      - 'recency_ratio': Fraction of total attribution in the final 20% of sequence tokens.
      - 'position_attributions': 1D tensor of attribution magnitude per sequence position.
      - 'early_context_blindness': True if early tokens receive exactly zero attribution.
      - 'issues': Human-readable notes and warnings.
  """
  model.eval()

  attr, _ = compute_integrated_gradients(
    model,
    sample_input,
    target=target,
    num_steps=num_steps,
    embedding_module=embedding_module,
  )

  # Aggregate attribution magnitude per position: (seq_len,)
  if attr.dim() == 2:  # (batch, seq_len)
    pos_attr = attr.abs().mean(dim=0)
  elif attr.dim() == 3:  # (batch, seq_len, hidden)
    pos_attr = attr.abs().sum(dim=-1).mean(dim=0)
  else:
    raise ValueError(f"Expected 2D or 3D sequence tensor, got shape {attr.shape}")

  seq_len = pos_attr.shape[0]
  if seq_len < 4:
    return {
      "status": "healthy",
      "recency_ratio": 0.5,
      "position_attributions": pos_attr,
      "early_context_blindness": False,
      "issues": ["Sequence length too short for context decay analysis."],
    }

  recent_idx = max(1, int(seq_len * 0.8))
  total_mass = pos_attr.sum().item() + 1e-8
  recent_mass = pos_attr[recent_idx:].sum().item()
  early_mass = pos_attr[:int(seq_len * 0.2)].sum().item()

  recency_ratio = recent_mass / total_mass
  early_blindness = early_mass < 1e-7

  status = "healthy"
  issues = []
  if early_blindness:
    status = "failed"
    issues.append(
      "Early context blindness detected: tokens in first 20% of sequence receive zero attribution."
    )
  elif recency_ratio > 0.95:
    status = "warning"
    issues.append(
      f"Severe recency bias ({recency_ratio:.1%} attribution concentrated in last 20% of tokens)."
    )

  return {
    "status": status,
    "recency_ratio": recency_ratio,
    "position_attributions": pos_attr,
    "early_context_blindness": early_blindness,
    "issues": issues,
  }
