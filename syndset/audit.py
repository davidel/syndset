"""Top-level architecture audit orchestrator."""

import torch.nn as nn

from syndset.diagnostics.capacity import check_overfit_capacity
from syndset.diagnostics.gradient import check_gradient_flow
from syndset.diagnostics.representation import check_dead_units, check_effective_rank
from syndset.diagnostics.stability import check_numerical_stability
from syndset.domains.llm import AssociativeRecallDataset
from syndset.domains.tabular import ConcentricHyperspheresDataset
from syndset.domains.timeseries import HarmonicSuperpositionDataset
from syndset.domains.vision import (
  SyntheticDetectionDataset,
  SyntheticSegmentationDataset,
  SyntheticShapesDataset,
)
from syndset.report import AuditReport


def audit(
  model,
  sample_input,
  sample_target=None,
  loss_fn=None,
  run_overfit=True,
  run_rank=True,
  run_stability=True,
  model_name=None,
):
  """Runs a comprehensive battery of diagnostic health checks on a neural architecture.

  Executes gradient flow verification, representation collapse checks,
  numerical stability tests, and single-batch capacity tests in seconds.
  Aggregates findings into an `AuditReport` scorecard.

  Args:
    model: The `torch.nn.Module` to evaluate.
    sample_input: Sample tensor or dict/tuple matching model forward inputs.
      Must have a batch dimension (e.g. shape `(B, ...)`).
    sample_target: Optional target tensor corresponding to `sample_input`.
      Required for overfit capacity probe when `run_overfit=True`.
    loss_fn: Optional loss function callable `loss_fn(outputs, targets)`. If `None`,
      defaults to cross-entropy for 1D integer targets or MSE loss otherwise.
    run_overfit: Whether to run single-batch memorization test (default: True).
    run_rank: Whether to compute effective rank and dead units (default: True).
    run_stability: Whether to check half precision (FP16/BF16) and NaN stability (default: True).
    model_name: Optional string name for the model in report summary. If `None`,
      uses `model.__class__.__name__`.

  Returns:
    An `AuditReport` instance containing detailed findings and scorecard.
    Call `report.print_summary()` to view findings or `report.is_healthy()` to inspect status.

  Example:
    >>> import torch
    >>> import torch.nn as nn
    >>> import syndset as syn
    >>> model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    >>> inputs = torch.randn(8, 16)
    >>> targets = torch.randint(0, 2, (8,))
    >>> report = syn.audit(model, inputs, targets, model_name="MLP")
    >>> report.print_summary()
  """
  name = model_name or model.__class__.__name__
  report = AuditReport(model_name=name)

  # 1. Gradient Flow Probe
  grad_res = check_gradient_flow(model, sample_input, target=sample_target, loss_fn=loss_fn)
  num_zeros = len(grad_res["zero_grad_params"])
  summary_text = (f"Norm: {grad_res['global_norm']:.3f}, "
                  f"Vanishing ratio: {grad_res['vanishing_ratio']:.2e}, "
                  f"Zero-grad params: {num_zeros}")
  report.add_check("Gradient Flow", grad_res["status"], summary_text, grad_res["issues"])

  # 2. Representation & Rank Probes
  if run_rank:
    rank_res = check_effective_rank(model, sample_input)
    num_collapsed = len(rank_res["collapsed_layers"])
    rank_summary = f"{len(rank_res['layer_ranks'])} layers monitored, {num_collapsed} collapsed"
    report.add_check("Effective Rank", rank_res["status"], rank_summary, rank_res["issues"])

    dead_res = check_dead_units(model, sample_input)
    num_dead_layers = len(dead_res["problematic_layers"])
    dead_summary = f"{num_dead_layers} layers with >30% dead units"
    report.add_check("Dead Units", dead_res["status"], dead_summary, dead_res["issues"])

  # 3. Numerical Stability Probe
  if run_stability:
    stab_res = check_numerical_stability(model, sample_input)
    half_err = stab_res["half_precision_relative_error"]
    err_str = f"{half_err:.2%}" if half_err is not None else "N/A"
    stab_summary = f"Max act: {stab_res['max_abs_activation']:.2f}, Half diff: {err_str}"
    report.add_check("Numerical Stability", stab_res["status"], stab_summary, stab_res["issues"])

  # 4. Capacity & Single-Batch Overfit Probe
  if run_overfit and sample_target is not None:
    cap_res = check_overfit_capacity(model, sample_input, sample_target, loss_fn=loss_fn)
    steps_val = cap_res["steps_to_converge"]
    steps_str = f"in {steps_val} steps" if cap_res["converged"] else "not converged"
    cap_summary = f"Final loss: {cap_res['final_loss']:.4f} ({steps_str})"
    report.add_check("Overfit Capacity", cap_res["status"], cap_summary, cap_res["issues"])

  return report


def audit_llm(model, num_pairs=8, vocab_size=64, batch_size=8, seed=42):
  """Audits an autoregressive or sequence model on synthetic associative recall.

  Instantiates an `AssociativeRecallDataset` and runs diagnostic probes (gradient flow,
  effective rank, dead units, numerical stability, and single-batch capacity).

  Args:
    model: The sequence `torch.nn.Module` to evaluate. Must accept input token ID
      tensors of shape `(batch_size, 2 * num_pairs + 1)` with dtype `torch.long`.
    num_pairs: Number of distinct (key, value) pairs per sequence (default: 8).
    vocab_size: Total vocabulary size partitioned between keys and values (default: 64).
      Must satisfy `vocab_size >= 2 * num_pairs`.
    batch_size: Batch size of synthetic sequences to audit (default: 8).
    seed: Random seed for reproducible generation (default: 42).

  Returns:
    An `AuditReport` summarizing model health on the sequence task.

  Example:
    >>> import torch.nn as nn
    >>> import syndset as syn
    >>> class SimpleSeqModel(nn.Module):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.emb = nn.Embedding(64, 32)
    ...     self.gru = nn.GRU(32, 32, batch_first=True)
    ...     self.head = nn.Linear(32, 64)
    ...   def forward(self, x):
    ...     out, _ = self.gru(self.emb(x))
    ...     return self.head(out[:, -1, :])
    >>> model = SimpleSeqModel()
    >>> report = syn.audit_llm(model, num_pairs=8, vocab_size=64)
    >>> report.print_summary()
  """
  dataset = AssociativeRecallDataset(num_samples=batch_size,
                                     num_pairs=num_pairs,
                                     vocab_size=vocab_size,
                                     seed=seed)
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (LLM Task)")


def audit_vision(model, img_size=32, channels=1, num_classes=None, batch_size=8, seed=42):
  """Audits a vision classifier architecture on procedural geometric shapes.

  Instantiates a `SyntheticShapesDataset` across geometric classes (circles,
  squares, triangles, crosses, rectangles, ellipses, annuli, diamonds) and executes
  gradient flow, representation rank, numerical stability, and overfit probes.

  Args:
    model: The vision `torch.nn.Module` to evaluate. Accepts image tensors of shape
      `(batch_size, channels, img_size, img_size)` and outputs logits of shape
      `(batch_size, num_classes)`.
    img_size: Image height and width in pixels (default: 32).
    channels: Image channels (default: 1).
    num_classes: Number of shape classes in `[1, 8]`. If `None`, attempts to infer
      from `model.num_classes` or `model.classifier.out_features`, defaulting to 8.
    batch_size: Batch size for audit (default: 8).
    seed: Random seed for reproducible generation (default: 42).

  Returns:
    An `AuditReport` summarizing model health on synthetic vision inputs.

  Example:
    >>> import torch.nn as nn
    >>> import syndset as syn
    >>> model = nn.Sequential(
    ...   nn.Conv2d(1, 16, 3, padding=1),
    ...   nn.ReLU(),
    ...   nn.AdaptiveAvgPool2d((1, 1)),
    ...   nn.Flatten(),
    ...   nn.Linear(16, 8)
    ... )
    >>> report = syn.audit_vision(model, img_size=32, channels=1, num_classes=8)
    >>> report.print_summary()
  """
  if num_classes is None:
    num_classes = getattr(model, "num_classes", None)
    classifier = getattr(model, "classifier", None)
    if num_classes is None and classifier is not None and hasattr(classifier, "out_features"):
      num_classes = classifier.out_features
    if num_classes is None:
      num_classes = 8

  dataset = SyntheticShapesDataset(num_samples=batch_size,
                                   img_size=img_size,
                                   channels=channels,
                                   num_classes=num_classes,
                                   seed=seed)
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (Vision Task)")


def audit_tabular(model, dim=16, num_classes=3, batch_size=16, seed=42):
  """Audits a feedforward / tabular architecture on concentric hyperspheres.

  Instantiates a `ConcentricHyperspheresDataset` to evaluate non-linear representation
  capacity, gradient flow, and numerical stability on non-linearly separable manifold shells.

  Args:
    model: The tabular `torch.nn.Module` to evaluate. Accepts float feature tensors
      of shape `(batch_size, dim)` and outputs logits of shape `(batch_size, num_classes)`.
    dim: Dimensionality of input feature vectors (default: 16).
    num_classes: Number of concentric spherical shells (default: 3).
    batch_size: Batch size for audit (default: 16).
    seed: Random seed for reproducible generation (default: 42).

  Returns:
    An `AuditReport` summarizing model health on non-linear manifold inputs.

  Example:
    >>> import torch.nn as nn
    >>> import syndset as syn
    >>> model = nn.Sequential(
    ...   nn.Linear(16, 32),
    ...   nn.ReLU(),
    ...   nn.Linear(32, 3)
    ... )
    >>> report = syn.audit_tabular(model, dim=16, num_classes=3)
    >>> report.print_summary()
  """
  dataset = ConcentricHyperspheresDataset(num_samples=batch_size,
                                          num_classes=num_classes,
                                          dim=dim,
                                          seed=seed)
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (Tabular Task)")


def audit_timeseries(model, seq_len=32, batch_size=8, seed=42):
  """Audits a time series or recurrent architecture on harmonic waveforms.

  Instantiates a `HarmonicSuperpositionDataset` with multi-frequency Fourier sums
  and checks whether the model preserves gradient flow, avoids representation collapse,
  and fits temporal sequences.

  Args:
    model: The time-series `torch.nn.Module` to evaluate. Accepts input tensors
      of shape `(batch_size, seq_len - 1, 1)` and outputs forecasted tensors of
    seq_len: Number of time steps in generated waveforms (default: 32).
      Model input sequence length is `seq_len - 1`.
    batch_size: Batch size for audit (default: 8).
    seed: Random seed for reproducible generation (default: 42).

  Returns:
    An `AuditReport` summarizing model health on spectral wave inputs.

  Example:
    >>> import torch.nn as nn
    >>> import syndset as syn
    >>> class SimpleForecaster(nn.Module):
    ...   def __init__(self):
    ...     super().__init__()
    ...     self.rnn = nn.GRU(1, 16, batch_first=True)
    ...     self.head = nn.Linear(16, 1)
    ...   def forward(self, x):
    ...     out, _ = self.rnn(x)
    ...     return self.head(out)
    >>> model = SimpleForecaster()
    >>> report = syn.audit_timeseries(model, seq_len=32)
    >>> report.print_summary()
  """
  dataset = HarmonicSuperpositionDataset(num_samples=batch_size, seq_len=seq_len, seed=seed)
  inputs, targets = dataset[:batch_size]
  return audit(model, inputs, targets, model_name=f"{model.__class__.__name__} (Time Series Task)")


def audit_segmentation(
  model,
  img_size=32,
  channels=1,
  num_classes=None,
  num_objects=3,
  batch_size=8,
  seed=42,
):
  """Audits a dense segmentation model (e.g. U-Net, FPN) on procedural pixel masks.

  Instantiates a `SyntheticSegmentationDataset` and evaluates encoder-decoder gradient
  propagation, representation rank across bottleneck and skip layers, numerical stability,
  and single-batch pixel-level overfit capacity using `nn.CrossEntropyLoss`.

  Args:
    model: The segmentation `torch.nn.Module` to evaluate. Must accept inputs of shape
      `(batch_size, channels, img_size, img_size)` and output logits of shape
      `(batch_size, num_classes + 1, img_size, img_size)`.
    img_size: Image height and width in pixels (default: 32).
    num_classes: Number of foreground shape classes in `[1, 8]`
      (default: `model.out_channels - 1` or 8).
      Total segmentation classes will be `num_classes + 1` (class 0 is background).
    num_objects: Number of shapes placed per image canvas (default: 3).
    batch_size: Batch size for audit (default: 8).
    seed: Random seed for reproducible generation (default: 42).

  Returns:
    An `AuditReport` summarizing model health on the semantic segmentation task.

  Example:
    >>> import torch.nn as nn
    >>> import syndset as syn
    >>> class SimpleUNet(nn.Module):
    ...   def __init__(self, in_c=1, out_c=9):
    ...     super().__init__()
    ...     self.enc = nn.Conv2d(in_c, 16, 3, padding=1)
    ...     self.dec = nn.Conv2d(16, out_c, 3, padding=1)
    ...   def forward(self, x):
    ...     return self.dec(nn.functional.relu(self.enc(x)))
    >>> model = SimpleUNet()
    >>> report = syn.audit_segmentation(model, img_size=32, channels=1, num_classes=8)
    >>> report.print_summary()
  """
  if num_classes is None:
    num_classes = getattr(model, "num_classes", None)
    if num_classes is None:
      out_channels = getattr(model, "out_channels", None)
      if out_channels is not None:
        num_classes = max(1, out_channels - 1)
    if num_classes is None:
      num_classes = 8

  dataset = SyntheticSegmentationDataset(
    num_samples=batch_size,
    img_size=img_size,
    channels=channels,
    num_classes=num_classes,
    num_objects=num_objects,
    seed=seed,
  )
  inputs, targets = dataset[:batch_size]
  return audit(
    model,
    inputs,
    targets,
    loss_fn=nn.CrossEntropyLoss(),
    model_name=f"{model.__class__.__name__} (Segmentation Task)",
  )


def audit_detection(
  model,
  img_size=32,
  channels=1,
  num_objects=3,
  num_classes=8,
  batch_size=8,
  loss_fn=None,
  seed=42,
):
  """Audits an object detection or coordinate regression model on procedural boxes.

  Instantiates a `SyntheticDetectionDataset` and evaluates coordinate regression
  gradient propagation, representation rank across backbone/pyramid layers, numerical
  stability, and single-batch bounding box overfit capacity.

  Args:
    model: The detector `torch.nn.Module` to evaluate. Accepts input tensors of shape
      `(batch_size, channels, img_size, img_size)` and must output predicted bounding
      boxes of shape `(batch_size, num_objects, 4)` in normalized `[cx, cy, w, h]` format
      (or tuple/dict containing box predictions).
    img_size: Image height and width in pixels (default: 32).
    channels: Image channels (default: 1).
    num_objects: Number of objects per image (default: 3).
    num_classes: Number of geometric shape classes in `[1, 8]` (default: 8).
    batch_size: Batch size for audit (default: 8).
    loss_fn: Optional loss function callable `loss_fn(preds, targets)`. If `None`,
      defaults to Smooth L1 loss on normalized box coordinates.
    seed: Random seed for reproducible generation (default: 42).

  Returns:
    An `AuditReport` summarizing model health on the object detection task.

  Example:
    >>> import torch.nn as nn
    >>> import syndset as syn
    >>> class SimpleDetector(nn.Module):
    ...   def __init__(self, num_objects=3):
    ...     super().__init__()
    ...     self.conv = nn.Conv2d(1, 16, 3, padding=1)
    ...     self.pool = nn.AdaptiveAvgPool2d((1, 1))
    ...     self.head = nn.Linear(16, num_objects * 4)
    ...     self.num_objects = num_objects
    ...   def forward(self, x):
    ...     h = self.pool(nn.functional.relu(self.conv(x))).flatten(1)
    ...     return torch.sigmoid(self.head(h)).view(-1, self.num_objects, 4)
    >>> model = SimpleDetector()
    >>> report = syn.audit_detection(model, img_size=32, channels=1, num_objects=3)
    >>> report.print_summary()
  """
  dataset = SyntheticDetectionDataset(
    num_samples=batch_size,
    img_size=img_size,
    channels=channels,
    num_classes=num_classes,
    num_objects=num_objects,
    seed=seed,
  )
  inputs = dataset.images[:batch_size]
  target_boxes = dataset.boxes[:batch_size]

  if loss_fn is None:

    def default_detection_loss(preds, targets):
      if isinstance(preds, (tuple, list)):
        preds = preds[0]
      elif isinstance(preds, dict):
        preds = preds.get("boxes", preds.get("pred_boxes", next(iter(preds.values()))))
      if preds.dim() == 2 and preds.shape[1] == targets.numel() // targets.shape[0]:
        preds = preds.view_as(targets)
      return nn.functional.smooth_l1_loss(preds, targets)

    loss_fn = default_detection_loss

  return audit(
    model,
    inputs,
    target_boxes,
    loss_fn=loss_fn,
    model_name=f"{model.__class__.__name__} (Detection Task)",
  )
