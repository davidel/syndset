"""Synthetic vision datasets for testing spatial inductive biases and convolutions."""

import math

import torch

from syndset.domains.base import SyntheticDataset

SHAPE_CIRCLE = 0
SHAPE_SQUARE = 1
SHAPE_TRIANGLE = 2
SHAPE_CROSS = 3
SHAPE_RECTANGLE = 4
SHAPE_ELLIPSE = 5
SHAPE_ANNULUS = 6
SHAPE_DIAMOND = 7

SHAPE_NAMES = [
  "circle",
  "square",
  "triangle",
  "cross",
  "rectangle",
  "ellipse",
  "annulus",
  "diamond",
]


def rasterize_circle(x_coords, y_coords, cx, cy, radius):
  """Rasterizes a filled circle onto coordinate grids."""
  dx = x_coords - cx
  dy = y_coords - cy
  mask = (torch.sqrt(dx**2 + dy**2) <= radius).float()
  bbox = (cx, cy, 2.0 * radius, 2.0 * radius)
  return mask, bbox


def rasterize_square(x_coords, y_coords, cx, cy, radius):
  """Rasterizes an axis-aligned square onto coordinate grids."""
  dx = x_coords - cx
  dy = y_coords - cy
  mask = ((dx.abs() <= radius) & (dy.abs() <= radius)).float()
  bbox = (cx, cy, 2.0 * radius, 2.0 * radius)
  return mask, bbox


def rasterize_triangle(x_coords, y_coords, cx, cy, radius):
  """Rasterizes an upward-pointing triangle onto coordinate grids."""
  dx = x_coords - cx
  dy = y_coords - cy
  in_height = (dy >= -radius) & (dy <= radius)
  in_width = dx.abs() <= (radius - dy * 0.5)
  mask = (in_height & in_width).float()
  bbox = (cx, cy, 2.0 * radius, 2.0 * radius)
  return mask, bbox


def rasterize_cross(x_coords, y_coords, cx, cy, radius, arm_ratio=0.35):
  """Rasterizes an orthogonal plus / cross shape onto coordinate grids."""
  dx = x_coords - cx
  dy = y_coords - cy
  arm_w = radius * arm_ratio
  horiz = (dy.abs() <= arm_w) & (dx.abs() <= radius)
  vert = (dx.abs() <= arm_w) & (dy.abs() <= radius)
  mask = (horiz | vert).float()
  bbox = (cx, cy, 2.0 * radius, 2.0 * radius)
  return mask, bbox


def rasterize_rectangle(x_coords, y_coords, cx, cy, rx, ry):
  """Rasterizes an axis-aligned rectangle with semi-axes rx and ry."""
  dx = x_coords - cx
  dy = y_coords - cy
  mask = ((dx.abs() <= rx) & (dy.abs() <= ry)).float()
  bbox = (cx, cy, 2.0 * rx, 2.0 * ry)
  return mask, bbox


def rasterize_ellipse(x_coords, y_coords, cx, cy, rx, ry):
  """Rasterizes a smooth ellipse with semi-axes rx and ry."""
  dx = x_coords - cx
  dy = y_coords - cy
  mask = (((dx / rx)**2 + (dy / ry)**2) <= 1.0).float()
  bbox = (cx, cy, 2.0 * rx, 2.0 * ry)
  return mask, bbox


def rasterize_annulus(x_coords, y_coords, cx, cy, r_inner, r_outer):
  """Rasterizes a ring / annulus (non-simply-connected topology with central hole)."""
  dx = x_coords - cx
  dy = y_coords - cy
  dist_sq = dx**2 + dy**2
  mask = ((dist_sq >= r_inner**2) & (dist_sq <= r_outer**2)).float()
  bbox = (cx, cy, 2.0 * r_outer, 2.0 * r_outer)
  return mask, bbox


def rasterize_diamond(x_coords, y_coords, cx, cy, radius):
  """Rasterizes a diamond / rhombus (L1 metric ball rotated 45 degrees)."""
  dx = x_coords - cx
  dy = y_coords - cy
  r_diamond = radius * 1.15
  mask = (dx.abs() + dy.abs() <= r_diamond).float()
  bbox = (cx, cy, 2.0 * r_diamond, 2.0 * r_diamond)
  return mask, bbox


def rasterize_shape(
  shape_id,
  x_coords,
  y_coords,
  cx,
  cy,
  radius,
  aspect_ratio=1.5,
  inner_ratio=0.45,
):
  """Dispatches shape rasterization by shape ID.

  Args:
    shape_id: Integer in [0, 7].
    x_coords: 2D meshgrid tensor of X coordinates.
    y_coords: 2D meshgrid tensor of Y coordinates.
    cx: Center X coordinate.
    cy: Center Y coordinate.
    radius: Scale radius parameter.
    aspect_ratio: Aspect ratio for rectangle and ellipse semi-axes.
    inner_ratio: Inner radius fraction for annulus.

  Returns:
    Tuple of (mask_tensor, bbox_tuple) where mask_tensor is (H, W) float
    and bbox_tuple is (cx, cy, width, height).
  """
  if shape_id == SHAPE_CIRCLE:
    return rasterize_circle(x_coords, y_coords, cx, cy, radius)
  elif shape_id == SHAPE_SQUARE:
    return rasterize_square(x_coords, y_coords, cx, cy, radius)
  elif shape_id == SHAPE_TRIANGLE:
    return rasterize_triangle(x_coords, y_coords, cx, cy, radius)
  elif shape_id == SHAPE_CROSS:
    return rasterize_cross(x_coords, y_coords, cx, cy, radius)
  elif shape_id == SHAPE_RECTANGLE:
    rx = radius * math.sqrt(aspect_ratio)
    ry = radius / math.sqrt(aspect_ratio)
    return rasterize_rectangle(x_coords, y_coords, cx, cy, rx, ry)
  elif shape_id == SHAPE_ELLIPSE:
    rx = radius * math.sqrt(aspect_ratio)
    ry = radius / math.sqrt(aspect_ratio)
    return rasterize_ellipse(x_coords, y_coords, cx, cy, rx, ry)
  elif shape_id == SHAPE_ANNULUS:
    r_inner = radius * inner_ratio
    return rasterize_annulus(x_coords, y_coords, cx, cy, r_inner, radius)
  elif shape_id == SHAPE_DIAMOND:
    return rasterize_diamond(x_coords, y_coords, cx, cy, radius)
  else:
    raise ValueError(f"Unknown shape_id {shape_id}. Must be in [0, {len(SHAPE_NAMES) - 1}].")


class SyntheticShapesDataset(SyntheticDataset):
  """Procedurally generates geometric shapes across 8 distinct geometric classes.

  Supports circles, squares, triangles, pluses, rectangles, ellipses,
  annuli (rings with holes), and diamonds. Provides a clean, controllable
  testbed for vision architectures (CNNs, ViTs, MLP-Mixers) to test spatial pattern
  recognition, translation invariance, aspect ratio robustness, and background
  noise robustness without needing heavy external image datasets.

  What this dataset generates:
    1. Continuous Coordinate Grid:
       - Constructs a 2D meshgrid of coordinates (x, y) spanning [-1.0, 1.0] x [-1.0, 1.0]
         over an `img_size x img_size` raster canvas.
    2. Geometric Shape Synthesis:
       - Samples a class label uniformly from {0, ..., num_classes - 1}:
         * Class 0 (Circle): dx^2 + dy^2 <= r^2
         * Class 1 (Square): max(|dx|, |dy|) <= r
         * Class 2 (Triangle): upward-pointing isosceles triangle of height 2r
         * Class 3 (Cross): orthogonal horizontal and vertical bars with width 0.35r
         * Class 4 (Rectangle): axis-aligned box with aspect ratio 1.8 or 0.55
         * Class 5 (Ellipse): (dx/rx)^2 + (dy/ry)^2 <= 1.0
         * Class 6 (Annulus): topological ring with central hole (0.45r <= dist <= r)
         * Class 7 (Diamond): L1 metric ball |dx| + |dy| <= 1.15r
       - Samples center (cx, cy) randomly in [-0.3, 0.3] (guaranteeing shape remains in-frame).
       - Samples radius uniformly in [0.2, 0.4].
    3. Noise & Channel Formatting:
       - Computes analytic foreground binary mask in {0.0, 1.0}.
       - Adds additive Gaussian background noise N(0, noise_std^2) and clamps pixel values
         to [0.0, 1.0].
       - Expands tensor across `channels` dimension (e.g. 1 for grayscale, 3 for replicated RGB).
    4. Target Output:
       - The target is the scalar integer class index in [0, num_classes - 1].

  Tensor Layout:
    images[idx]: 3D tensor of shape `(channels, img_size, img_size)` (float32 in [0.0, 1.0])
    labels[idx]: scalar tensor with class index in [0, num_classes - 1] (int64)

  Concrete Example:
    Suppose img_size=32, channels=1, num_classes=8:
      Sample generated: Class 0 (Circle), center=(0.10, -0.05), radius=0.25
      image: tensor of shape (1, 32, 32) with a smooth circular white patch on noisy dark background
      label: tensor(0)

  Attributes:
    images: Tensor of shape `(num_samples, channels, img_size, img_size)`
      with dtype `torch.float32`.
    labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `[0, num_classes - 1]`.

  Item Format:
    Indexing `dataset[idx]` returns `(image, label)` where:
      - `image`: 3D `torch.float32` tensor of shape `(channels, img_size, img_size)`
        with pixel values in `[0.0, 1.0]`.
      - `label`: scalar `torch.long` tensor with shape class index in `[0, num_classes - 1]`.
  """

  def __init__(
    self,
    num_samples=1000,
    img_size=32,
    channels=1,
    noise_std=0.1,
    num_classes=8,
    seed=42,
  ):
    """Initializes the synthetic shapes dataset.

    Args:
      num_samples: Number of images to generate (default: 1000).
      img_size: Height and width in pixels (default: 32).
      channels: Number of color channels (default: 1).
      noise_std: Standard deviation of background Gaussian noise (default: 0.1).
      num_classes: Number of geometric shape classes, between 1 and 8 (default: 8).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    if not (1 <= num_classes <= len(SHAPE_NAMES)):
      raise ValueError(f"num_classes must be between 1 and {len(SHAPE_NAMES)}, got {num_classes}")

    self._img_size = img_size
    self._channels = channels
    self._noise_std = noise_std
    self._num_classes = num_classes

    generator = torch.Generator().manual_seed(seed)
    images_list = []
    labels_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    for _ in range(num_samples):
      label = torch.randint(0, num_classes, (1,), generator=generator).item()
      cx = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
      cy = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
      radius = 0.2 + torch.rand(1, generator=generator).item() * 0.2

      aspect_ratio = 1.8 if torch.rand(1, generator=generator).item() > 0.5 else 0.55
      mask, _ = rasterize_shape(
        label,
        x_coords,
        y_coords,
        cx,
        cy,
        radius,
        aspect_ratio=aspect_ratio,
      )

      # Add noise and replicate across channels
      noise = torch.randn((img_size, img_size), generator=generator) * noise_std
      img = (mask + noise).clamp(0.0, 1.0)
      img = img.unsqueeze(0).repeat(channels, 1, 1)

      images_list.append(img)
      labels_list.append(label)

    self._images = torch.stack(images_list, dim=0)
    self._labels = torch.tensor(labels_list, dtype=torch.long)

  @property
  def img_size(self):
    """Returns the image dimension (pixels)."""
    return self._img_size

  @property
  def channels(self):
    """Returns the number of image channels."""
    return self._channels

  @property
  def noise_std(self):
    """Returns the background noise standard deviation."""
    return self._noise_std

  @property
  def num_classes(self):
    """Returns the number of geometric shape classes."""
    return self._num_classes

  @property
  def images(self):
    """Returns the tensor of generated images."""
    return self._images

  @property
  def labels(self):
    """Returns the tensor of shape class labels."""
    return self._labels

  def __getitem__(self, idx):
    """Returns an image tensor and class label."""
    return self._images[idx], self._labels[idx]

  def description(self):
    """Returns a description of the dataset."""
    dims_str = f"{self._channels}x{self._img_size}x{self._img_size}"
    return f"Synthetic Shapes ({self._num_classes} classes, {dims_str})"


class SyntheticSegmentationDataset(SyntheticDataset):
  """Procedurally generates multi-class semantic segmentation images and pixel masks.

  Generates 2D images containing multiple geometric shapes rasterized onto a shared
  canvas, accompanied by pixel-wise ground-truth class label masks. Background is
  labeled as class 0, and foreground shapes correspond to classes 1 through num_classes.

  Provides a clean, fast testbed for dense prediction architectures (U-Net, FPN,
  DeepLab, SegFormer) to evaluate spatial detail retention, encoder-decoder skip
  connection gradients, and boundary segmentation without heavy external datasets.

  What this dataset generates:
    1. Multi-Object Canvas Construction:
       - Initializes an empty canvas of size `img_size x img_size` with background 0.
       - Initializes an integer semantic mask tensor of zeros (class 0: background).
    2. Sequential Object Rasterization:
       - Sequentially places `num_objects` shapes onto the canvas.
       - For each object:
         * Samples shape ID in {0, ..., num_classes - 1}.
         * Samples random center (cx, cy) and radius.
         * Rasterizes analytic shape indicator.
         * Updates image canvas with shape pixel values.
         * Overlays pixel mask: all pixels covered by this shape receive class label
           `shape_id + 1` (integer values in [1, num_classes]).
    3. Noise & Channel Formatting:
       - Adds Gaussian noise N(0, noise_std^2) to the image canvas, clamped to [0.0, 1.0].
       - Expands to `channels` (shape: (channels, img_size, img_size)).
    4. Target Output:
       - The target is the 2D integer tensor of pixel-wise semantic class IDs
         (shape: (img_size, img_size)).

  Tensor Layout:
    images[idx]: 3D tensor of shape `(channels, img_size, img_size)` (float32 in [0.0, 1.0])
    masks[idx]:  2D tensor of shape `(img_size, img_size)` (int64 in 0..num_classes)

  Concrete Example:
    Suppose img_size=32, num_objects=2, num_classes=3 (1: circle, 2: square, 3: triangle):
      Image contains a circle and a triangle.
      mask[y, x] = 0 for background pixels
      mask[y, x] = 1 for pixels within the circle boundary
      mask[y, x] = 3 for pixels within the triangle boundary

  Attributes:
    images: Tensor of shape `(num_samples, channels, img_size, img_size)`
      with dtype `torch.float32`.
    masks: Tensor of shape `(num_samples, img_size, img_size)` with dtype `torch.long`.

  Item Format:
    Indexing `dataset[idx]` returns `(image, mask)` where:
      - `image`: 3D `torch.float32` tensor of shape `(channels, img_size, img_size)`
        with pixel values in `[0.0, 1.0]`.
      - `mask`: 2D `torch.long` tensor of shape `(img_size, img_size)` with pixel
        classes in `[0, num_classes]`.
  """

  def __init__(
    self,
    num_samples=1000,
    img_size=32,
    channels=1,
    num_classes=8,
    num_objects=3,
    noise_std=0.1,
    seed=42,
  ):
    """Initializes the synthetic segmentation dataset.

    Args:
      num_samples: Number of images to generate (default: 1000).
      img_size: Height and width in pixels (default: 32).
      channels: Number of color channels (default: 1).
      num_classes: Number of distinct foreground shape classes, between 1 and 8 (default: 8).
        Total classes in masks will be `num_classes + 1` (0 is background).
      num_objects: Number of shapes to place per image (default: 3).
      noise_std: Standard deviation of background Gaussian noise (default: 0.1).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    if not (1 <= num_classes <= len(SHAPE_NAMES)):
      raise ValueError(f"num_classes must be between 1 and {len(SHAPE_NAMES)}, got {num_classes}")
    if num_objects < 1:
      raise ValueError(f"num_objects must be >= 1, got {num_objects}")

    self._img_size = img_size
    self._channels = channels
    self._num_classes = num_classes
    self._num_objects = num_objects
    self._noise_std = noise_std

    generator = torch.Generator().manual_seed(seed)
    images_list = []
    masks_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    for _ in range(num_samples):
      canvas = torch.zeros((img_size, img_size), dtype=torch.float32)
      mask = torch.zeros((img_size, img_size), dtype=torch.long)

      for _ in range(num_objects):
        shape_id = torch.randint(0, num_classes, (1,), generator=generator).item()
        cx = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
        cy = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
        radius = 0.15 + torch.rand(1, generator=generator).item() * 0.15
        aspect_ratio = 1.6 if torch.rand(1, generator=generator).item() > 0.5 else 0.6

        shape_mask, _ = rasterize_shape(
          shape_id,
          x_coords,
          y_coords,
          cx,
          cy,
          radius,
          aspect_ratio=aspect_ratio,
        )

        foreground = shape_mask > 0.5
        canvas = torch.where(foreground, shape_mask, canvas)
        mask = torch.where(foreground, shape_id + 1, mask)

      noise = torch.randn((img_size, img_size), generator=generator) * noise_std
      img = (canvas + noise).clamp(0.0, 1.0)
      img = img.unsqueeze(0).repeat(channels, 1, 1)

      images_list.append(img)
      masks_list.append(mask)

    self._images = torch.stack(images_list, dim=0)
    self._masks = torch.stack(masks_list, dim=0)

  @property
  def img_size(self):
    """Returns the image resolution."""
    return self._img_size

  @property
  def channels(self):
    """Returns the number of image channels."""
    return self._channels

  @property
  def num_classes(self):
    """Returns the number of foreground shape classes."""
    return self._num_classes

  @property
  def num_classes_total(self):
    """Returns the total number of classes including background (class 0)."""
    return self._num_classes + 1

  @property
  def num_objects(self):
    """Returns the number of objects per sample image."""
    return self._num_objects

  @property
  def noise_std(self):
    """Returns the noise standard deviation."""
    return self._noise_std

  @property
  def images(self):
    """Returns the tensor of generated images."""
    return self._images

  @property
  def masks(self):
    """Returns the tensor of ground-truth segmentation masks."""
    return self._masks

  def __getitem__(self, idx):
    """Returns an image tensor and corresponding pixel class mask."""
    return self._images[idx], self._masks[idx]

  def description(self):
    """Returns a description of the segmentation dataset."""
    dims_str = f"{self._channels}x{self._img_size}x{self._img_size}"
    return f"Synthetic Semantic Segmentation ({self._num_classes + 1} classes, {dims_str})"


class SyntheticDetectionDataset(SyntheticDataset):
  """Procedurally generates object detection images, bounding boxes, and class labels.

  Places multiple procedural geometric shapes onto a shared canvas, accompanied by
  normalized bounding boxes in [cx, cy, w, h] format (values in [0.0, 1.0]) and class
  labels.

  Provides a clean, fast testbed for object detectors (Faster R-CNN, SSD, YOLO, DETR)
  to evaluate coordinate regression pathways, anchor assignment, scale invariance,
  and bipartite matching without requiring large image benchmarks.

  What this dataset generates:
    1. Multi-Object Scene Layout:
       - Renders `num_objects` procedural shapes onto an `(img_size, img_size)` canvas.
    2. Bounding Box Computation:
       - For each placed shape, computes the exact bounding box [cx, cy, w, h] from its
         analytic geometry parameters and maps from [-1.0, 1.0] to normalized [0.0, 1.0]
         coordinates.
    3. Output Tensors:
       - `images`: Tensor of shape `(num_samples, channels, img_size, img_size)`.
       - `boxes`: Tensor of shape `(num_samples, num_objects, 4)` in normalized `[cx, cy, w, h]`.
       - `labels`: Tensor of shape `(num_samples, num_objects)` with shape class IDs.

  Tensor Layout:
    images[idx]: 3D tensor of shape `(channels, img_size, img_size)` (float32 in [0.0, 1.0])
    boxes[idx]:  2D tensor of shape `(num_objects, 4)` (float32 in [0.0, 1.0])
    labels[idx]: 1D tensor of shape `(num_objects,)` (int64 in 0..num_classes - 1)

  Concrete Example:
    Suppose img_size=32, num_objects=2, num_classes=8:
      Object 0: Class 0 (Circle), normalized box=[0.35, 0.40, 0.20, 0.20]
      Object 1: Class 1 (Square), normalized box=[0.70, 0.65, 0.25, 0.25]
      boxes[0]:  [[0.35, 0.40, 0.20, 0.20], [0.70, 0.65, 0.25, 0.25]]
      labels[0]: [0, 1]

  Attributes:
    images: Tensor of shape `(num_samples, channels, img_size, img_size)`
      with dtype `torch.float32`.
    boxes: Tensor of shape `(num_samples, num_objects, 4)` with dtype `torch.float32`
      in normalized `[cx, cy, w, h]` format in `[0.0, 1.0]`.
    labels: Tensor of shape `(num_samples, num_objects)` with dtype `torch.long`
      in `[0, num_classes - 1]`.

  Item Format:
    Indexing `dataset[idx]` returns `(image, boxes, labels)` where:
      - `image`: 3D `torch.float32` tensor of shape `(channels, img_size, img_size)`
        in `[0.0, 1.0]`.
      - `boxes`: 2D `torch.float32` tensor of shape `(num_objects, 4)` with normalized coordinates.
      - `labels`: 1D `torch.long` tensor of shape `(num_objects,)` with shape class IDs.
  """

  def __init__(
    self,
    num_samples=1000,
    img_size=32,
    channels=1,
    num_classes=8,
    num_objects=3,
    noise_std=0.1,
    seed=42,
  ):
    """Initializes the synthetic detection dataset.

    Args:
      num_samples: Number of images to generate (default: 1000).
      img_size: Height and width in pixels (default: 32).
      channels: Number of color channels (default: 1).
      num_classes: Number of distinct shape classes, between 1 and 8 (default: 8).
      num_objects: Number of objects per image (default: 3).
      noise_std: Standard deviation of background Gaussian noise (default: 0.1).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    if not (1 <= num_classes <= len(SHAPE_NAMES)):
      raise ValueError(f"num_classes must be between 1 and {len(SHAPE_NAMES)}, got {num_classes}")
    if num_objects < 1:
      raise ValueError(f"num_objects must be >= 1, got {num_objects}")

    self._img_size = img_size
    self._channels = channels
    self._num_classes = num_classes
    self._num_objects = num_objects
    self._noise_std = noise_std

    generator = torch.Generator().manual_seed(seed)
    images_list = []
    boxes_list = []
    labels_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    for _ in range(num_samples):
      canvas = torch.zeros((img_size, img_size), dtype=torch.float32)
      sample_boxes = []
      sample_labels = []

      for _ in range(num_objects):
        shape_id = torch.randint(0, num_classes, (1,), generator=generator).item()
        cx = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
        cy = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
        radius = 0.15 + torch.rand(1, generator=generator).item() * 0.15
        aspect_ratio = 1.6 if torch.rand(1, generator=generator).item() > 0.5 else 0.6

        shape_mask, raw_bbox = rasterize_shape(
          shape_id,
          x_coords,
          y_coords,
          cx,
          cy,
          radius,
          aspect_ratio=aspect_ratio,
        )

        foreground = shape_mask > 0.5
        canvas = torch.where(foreground, shape_mask, canvas)

        # Convert [-1.0, 1.0] bbox coordinates to normalized [0.0, 1.0] range
        raw_cx, raw_cy, raw_w, raw_h = raw_bbox
        norm_cx = (raw_cx + 1.0) / 2.0
        norm_cy = (raw_cy + 1.0) / 2.0
        norm_w = raw_w / 2.0
        norm_h = raw_h / 2.0
        sample_boxes.append([norm_cx, norm_cy, norm_w, norm_h])
        sample_labels.append(shape_id)

      noise = torch.randn((img_size, img_size), generator=generator) * noise_std
      img = (canvas + noise).clamp(0.0, 1.0)
      img = img.unsqueeze(0).repeat(channels, 1, 1)

      images_list.append(img)
      boxes_list.append(torch.tensor(sample_boxes, dtype=torch.float32))
      labels_list.append(torch.tensor(sample_labels, dtype=torch.long))

    self._images = torch.stack(images_list, dim=0)
    self._boxes = torch.stack(boxes_list, dim=0)
    self._labels = torch.stack(labels_list, dim=0)

  @property
  def img_size(self):
    """Returns the image resolution."""
    return self._img_size

  @property
  def channels(self):
    """Returns the number of image channels."""
    return self._channels

  @property
  def num_classes(self):
    """Returns the number of geometric shape classes."""
    return self._num_classes

  @property
  def num_objects(self):
    """Returns the number of objects per sample image."""
    return self._num_objects

  @property
  def noise_std(self):
    """Returns the noise standard deviation."""
    return self._noise_std

  @property
  def images(self):
    """Returns the tensor of generated images."""
    return self._images

  @property
  def boxes(self):
    """Returns the tensor of normalized bounding boxes [cx, cy, w, h]."""
    return self._boxes

  @property
  def labels(self):
    """Returns the tensor of shape class labels."""
    return self._labels

  def __getitem__(self, idx):
    """Returns an image tensor, bounding boxes, and class labels."""
    return self._images[idx], self._boxes[idx], self._labels[idx]

  def description(self):
    """Returns a description of the detection dataset."""
    dims_str = f"{self._channels}x{self._img_size}x{self._img_size}"
    return (f"Synthetic Object Detection ({self._num_objects} objects, "
            f"{self._num_classes} classes, {dims_str})")


class NumerosityDataset(SyntheticDataset):
  """Evaluates an architecture's capacity for discrete entity counting (numerosity).

  Procedurally renders N distinct non-overlapping geometric shapes (dots / circles)
  scattered across a canvas. The target is the discrete count integer label.

  Probes whether an architecture's pooling mechanism (Global Average Pooling vs.
  Attention Pooling / Class Token) can aggregate discrete instance counts without
  confounding count with total pixel area or intensity.

  What this dataset generates:
    1. Object Count Selection:
       - Uniformly samples an integer count k in [min_count, max_count].
    2. Collision-Free Radial Placement:
       - Places k non-overlapping circular dots of fixed radius on the canvas.
       - Uses rejection sampling with distance constraint dist(c_i, c_j) >= 2.4 * radius
         to prevent occlusion or merging.
    3. Noise & Channel Formatting:
       - Rasterizes circles onto the canvas, adds Gaussian noise, clamps to [0.0, 1.0].
    4. Target Output:
       - The target label is 0-indexed count offset: `k - min_count` (int64).

  Tensor Layout:
    images[idx]: 3D tensor of shape `(channels, img_size, img_size)` (float32 in [0.0, 1.0])
    labels[idx]: scalar tensor with count index `k - min_count` in
                 [0, max_count - min_count] (int64)

  Concrete Example:
    Suppose min_count=1, max_count=6, img_size=32:
      Image generated with 4 scattered non-touching dots
      target label: 4 - 1 = 3 (class 3 corresponds to count 4)

  Attributes:
    images: Tensor of shape `(num_samples, channels, img_size, img_size)`
      with dtype `torch.float32`.
    labels: Tensor of shape `(num_samples,)` with dtype `torch.long`
      in `[0, max_count - min_count]`.

  Item Format:
    Indexing `dataset[idx]` returns `(image, label)` where:
      - `image`: 3D `torch.float32` tensor of shape `(channels, img_size, img_size)`
        in `[0.0, 1.0]`.
      - `label`: scalar `torch.long` tensor with count class offset `count - min_count`.
  """

  def __init__(
    self,
    num_samples=1000,
    img_size=32,
    channels=1,
    min_count=1,
    max_count=6,
    radius=0.08,
    noise_std=0.1,
    seed=42,
  ):
    """Initializes the numerosity counting dataset.

    Args:
      num_samples: Number of images to generate (default: 1000).
      img_size: Height and width in pixels (default: 32).
      channels: Number of color channels (default: 1).
      min_count: Minimum object count, >= 1 (default: 1).
      max_count: Maximum object count (default: 6).
      radius: Radius of each discrete dot (default: 0.08).
      noise_std: Standard deviation of background Gaussian noise (default: 0.1).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    if min_count < 1 or max_count < min_count:
      raise ValueError(f"Invalid count range: min_count={min_count}, max_count={max_count}")

    self._img_size = img_size
    self._channels = channels
    self._min_count = min_count
    self._max_count = max_count
    self._radius = radius
    self._noise_std = noise_std
    self._num_classes = max_count - min_count + 1

    generator = torch.Generator().manual_seed(seed)
    images_list = []
    labels_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    for _ in range(num_samples):
      count = torch.randint(min_count, max_count + 1, (1,), generator=generator).item()
      canvas = torch.zeros((img_size, img_size), dtype=torch.float32)

      centers = []
      min_dist = radius * 2.4

      for _ in range(count):
        placed = False
        for _ in range(50):
          cx = (torch.rand(1, generator=generator).item() - 0.5) * 1.5
          cy = (torch.rand(1, generator=generator).item() - 0.5) * 1.5
          if all(math.hypot(cx - ex, cy - ey) >= min_dist for ex, ey in centers):
            centers.append((cx, cy))
            placed = True
            break
        if not placed:
          cx = (torch.rand(1, generator=generator).item() - 0.5) * 1.5
          cy = (torch.rand(1, generator=generator).item() - 0.5) * 1.5
          centers.append((cx, cy))

      for cx, cy in centers:
        mask, _ = rasterize_circle(x_coords, y_coords, cx, cy, radius)
        canvas = torch.maximum(canvas, mask)

      noise = torch.randn((img_size, img_size), generator=generator) * noise_std
      img = (canvas + noise).clamp(0.0, 1.0)
      img = img.unsqueeze(0).repeat(channels, 1, 1)

      images_list.append(img)
      labels_list.append(count - min_count)

    self._images = torch.stack(images_list, dim=0)
    self._labels = torch.tensor(labels_list, dtype=torch.long)

  @property
  def img_size(self):
    """Returns the image resolution."""
    return self._img_size

  @property
  def channels(self):
    """Returns the number of image channels."""
    return self._channels

  @property
  def min_count(self):
    """Returns the minimum object count."""
    return self._min_count

  @property
  def max_count(self):
    """Returns the maximum object count."""
    return self._max_count

  @property
  def num_classes(self):
    """Returns the number of distinct count target classes."""
    return self._num_classes

  @property
  def noise_std(self):
    """Returns the background noise standard deviation."""
    return self._noise_std

  @property
  def images(self):
    """Returns the tensor of generated images."""
    return self._images

  @property
  def labels(self):
    """Returns the tensor of 0-indexed count labels."""
    return self._labels

  def __getitem__(self, idx):
    """Returns an image tensor and count class label."""
    return self._images[idx], self._labels[idx]

  def description(self):
    """Returns a description of the dataset."""
    dims_str = f"{self._channels}x{self._img_size}x{self._img_size}"
    return f"Numerosity Counting ({self._num_classes} classes, {dims_str})"


class SpatialRelationDataset(SyntheticDataset):
  """Tests whether an architecture exhibits relational spatial reasoning.

  Generates images containing two distinct geometric shapes (a circle and a square)
  at random locations. The target is a binary classification label indicating whether
  a spatial relation holds (e.g. whether the circle is to the left of the square).

  Diagnoses whether non-local cross-patch self-attention (ViTs) or coordinate-aware
  convolutions (CoordConv) can bind pairwise spatial coordinates across distant regions.

  What this dataset generates:
    1. Two-Entity Scene Assembly:
       - Renders Shape 1 (Circle) at random position (cx1, cy1).
       - Renders Shape 2 (Square) at random position (cx2, cy2).
    2. Spatial Relation Evaluation:
       - If relation == 'left': target is 1 if cx1 < cx2 (circle is left of square), else 0.
       - If relation == 'above': target is 1 if cy1 < cy2 (circle is above square), else 0.
    3. Noise & Formatting:
       - Overlays both shapes onto canvas, adds noise, and clamps to [0.0, 1.0].
    4. Target Output:
       - Scalar binary label in {0, 1} (int64).

  Tensor Layout:
    images[idx]: 3D tensor of shape `(channels, img_size, img_size)` (float32 in [0.0, 1.0])
    labels[idx]: scalar binary tensor in {0, 1} (int64)

  Concrete Example:
    Suppose relation='left', img_size=32:
      Circle placed at cx1 = -0.40, cy1 = 0.00
      Square placed at cx2 =  0.30, cy2 = 0.20
      Since cx1 (-0.40) < cx2 (0.30), circle is to the left of the square.
      Target label: 1

  Attributes:
    images: Tensor of shape `(num_samples, channels, img_size, img_size)`
      with dtype `torch.float32`.
    labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `{0, 1}`.

  Item Format:
    Indexing `dataset[idx]` returns `(image, label)` where:
      - `image`: 3D `torch.float32` tensor of shape `(channels, img_size, img_size)`
        in `[0.0, 1.0]`.
      - `label`: scalar `torch.long` tensor with binary relation value
        (1 if relation holds, 0 otherwise).
  """

  def __init__(
    self,
    num_samples=1000,
    img_size=32,
    channels=1,
    relation="left",
    noise_std=0.1,
    seed=42,
  ):
    """Initializes the spatial relation dataset.

    Args:
      num_samples: Number of images to generate (default: 1000).
      img_size: Height and width in pixels (default: 32).
      channels: Number of color channels (default: 1).
      relation: Spatial relation to test ('left' for cx1 < cx2, 'above' for cy1 < cy2).
      noise_std: Standard deviation of background Gaussian noise (default: 0.1).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    if relation not in ("left", "above"):
      raise ValueError(f"relation must be 'left' or 'above', got {relation}")

    self._img_size = img_size
    self._channels = channels
    self._relation = relation
    self._noise_std = noise_std
    self._num_classes = 2

    generator = torch.Generator().manual_seed(seed)
    images_list = []
    labels_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    radius = 0.2

    for _ in range(num_samples):
      cx1 = (torch.rand(1, generator=generator).item() - 0.5) * 1.2
      cy1 = (torch.rand(1, generator=generator).item() - 0.5) * 1.2

      for _ in range(50):
        cx2 = (torch.rand(1, generator=generator).item() - 0.5) * 1.2
        cy2 = (torch.rand(1, generator=generator).item() - 0.5) * 1.2
        if math.hypot(cx1 - cx2, cy1 - cy2) >= radius * 2.5:
          break

      # Shape 1: Circle, Shape 2: Square
      mask1, _ = rasterize_circle(x_coords, y_coords, cx1, cy1, radius)
      mask2, _ = rasterize_square(x_coords, y_coords, cx2, cy2, radius)

      canvas = torch.maximum(mask1, mask2)
      noise = torch.randn((img_size, img_size), generator=generator) * noise_std
      img = (canvas + noise).clamp(0.0, 1.0)
      img = img.unsqueeze(0).repeat(channels, 1, 1)

      if relation == "left":
        target = 1 if cx1 < cx2 else 0
      else:
        target = 1 if cy1 < cy2 else 0

      images_list.append(img)
      labels_list.append(target)

    self._images = torch.stack(images_list, dim=0)
    self._labels = torch.tensor(labels_list, dtype=torch.long)

  @property
  def img_size(self):
    """Returns the image resolution."""
    return self._img_size

  @property
  def channels(self):
    """Returns the number of image channels."""
    return self._channels

  @property
  def relation(self):
    """Returns the spatial relation tested ('left' or 'above')."""
    return self._relation

  @property
  def num_classes(self):
    """Returns the number of binary classes (2)."""
    return self._num_classes

  @property
  def noise_std(self):
    """Returns the background noise standard deviation."""
    return self._noise_std

  @property
  def images(self):
    """Returns the tensor of generated images."""
    return self._images

  @property
  def labels(self):
    """Returns the tensor of binary relation labels."""
    return self._labels

  def __getitem__(self, idx):
    """Returns an image tensor and binary relation label."""
    return self._images[idx], self._labels[idx]

  def description(self):
    """Returns a description of the dataset."""
    dims_str = f"{self._channels}x{self._img_size}x{self._img_size}"
    return f"Spatial Relation Reasoning ({self._relation}, {dims_str})"


class TextureVsShapeDataset(SyntheticDataset):
  """Tests whether a vision model relies primarily on shape contours or surface textures.

  Produces samples where a primary shape (circle vs square) is overlaid with
  a conflicting high-frequency texture (horizontal stripes vs vertical stripes).
  Allows checking whether an architecture exhibits shape bias (typical of humans
  and ViTs) or texture bias (typical of standard CNNs).

  What this dataset generates:
    1. Shape Geometry:
       - Randomly samples shape identity: 0 for Circle (radius 0.6), 1 for Square (half-width 0.6).
    2. Orthogonal Texture Patterns:
       - Randomly samples texture identity: 0 for horizontal square-wave stripes (freq=8),
         1 for vertical square-wave stripes (freq=8).
    3. Composite Image Rendering:
       - Fills the interior of the shape mask with modulated texture intensity:
         img = shape_mask * (0.3 + 0.7 * texture_pattern).
       - Pixels outside the shape boundary are 0.0 (clean background).
    4. Dual Target Labels:
       - `.shape_labels`: class in {0, 1} (0: Circle, 1: Square).
       - `.texture_labels`: class in {0, 1} (0: Horizontal, 1: Vertical).

  Tensor Layout:
    images[idx]:         3D tensor of shape `(1, img_size, img_size)` (float32 in [0.0, 1.0])
    shape_labels[idx]:   scalar binary tensor in {0, 1} (int64)
    texture_labels[idx]: scalar binary tensor in {0, 1} (int64)

  Concrete Example:
    Suppose img_size=32:
      Sample generated: Shape 0 (Circle), Texture 1 (Vertical stripes)
      image: circular silhouette filled with alternating black and white vertical zebra stripes
      shape_label: 0, texture_label: 1

  Attributes:
    images: Tensor of shape `(num_samples, 1, img_size, img_size)` with dtype `torch.float32`.
    shape_labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `{0, 1}`.
    texture_labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `{0, 1}`.

  Item Format:
    Indexing `dataset[idx]` returns `(image, shape_label)` where:
      - `image`: 3D `torch.float32` tensor of shape `(1, img_size, img_size)` in `[0.0, 1.0]`.
      - `shape_label`: scalar `torch.long` tensor with shape identity in `{0, 1}`
        (0: circle, 1: square).
  """

  def __init__(self, num_samples=1000, img_size=32, seed=42):
    """Initializes the texture vs shape dataset.

    Args:
      num_samples: Total number of samples (default: 1000).
      img_size: Image resolution (default: 32).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._img_size = img_size

    generator = torch.Generator().manual_seed(seed)
    images_list = []
    shape_labels_list = []
    texture_labels_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    for _ in range(num_samples):
      shape_id = torch.randint(0, 2, (1,), generator=generator).item()
      texture_id = torch.randint(0, 2, (1,), generator=generator).item()

      # Shape mask: 0 = circle, 1 = square
      if shape_id == 0:
        shape_mask = (torch.sqrt(x_coords**2 + y_coords**2) <= 0.6).float()
      else:
        shape_mask = ((x_coords.abs() <= 0.6) & (y_coords.abs() <= 0.6)).float()

      # Texture pattern: 0 = horizontal stripes, 1 = vertical stripes
      freq = 8.0
      if texture_id == 0:
        texture_pattern = (torch.sin(y_coords * freq * math.pi) > 0.0).float()
      else:
        texture_pattern = (torch.sin(x_coords * freq * math.pi) > 0.0).float()

      # Modulate shape interior with texture
      img = shape_mask * (0.3 + 0.7 * texture_pattern)
      img = img.unsqueeze(0)

      images_list.append(img)
      shape_labels_list.append(shape_id)
      texture_labels_list.append(texture_id)

    self._images = torch.stack(images_list, dim=0)
    self._shape_labels = torch.tensor(shape_labels_list, dtype=torch.long)
    self._texture_labels = torch.tensor(texture_labels_list, dtype=torch.long)

  @property
  def img_size(self):
    """Returns the image resolution."""
    return self._img_size

  @property
  def images(self):
    """Returns the tensor of generated images."""
    return self._images

  @property
  def shape_labels(self):
    """Returns the tensor of shape identity labels."""
    return self._shape_labels

  @property
  def texture_labels(self):
    """Returns the tensor of texture identity labels."""
    return self._texture_labels

  def __getitem__(self, idx):
    """Returns the image tensor and the shape label by default."""
    return self._images[idx], self._shape_labels[idx]

  def description(self):
    """Returns a description of the dataset."""
    return f"Texture vs Shape Conflict ({self._img_size}x{self._img_size})"


class SpatialInvarianceDataset(SyntheticDataset):
  """Tests whether model representations are invariant under spatial shifts.

  Generates pairs of base and shifted images containing identical shapes.
  Can be used to evaluate whether convolutional or attention layers preserve
  prediction stability under coordinate translations.

  What this dataset generates:
    1. Base Reference Image:
       - Renders a centered geometric shape (0: Circle, 1: Square) at (cx=0, cy=0) with radius 0.35.
    2. Translated Shift Image:
       - Renders the identical shape shifted horizontally along the X-axis by `shift_pixels`:
         shift_norm = (2.0 / img_size) * shift_pixels.
    3. Output Image Pairs:
       - `.base_images`: Tensor of shape `(num_samples, 1, img_size, img_size)`.
       - `.shifted_images`: Tensor of shape `(num_samples, 1, img_size, img_size)`.
       - `.labels`: Tensor of shape `(num_samples,)` in {0, 1}.

  Tensor Layout:
    base_images[idx]:    3D tensor of shape `(1, img_size, img_size)`
    shifted_images[idx]: 3D tensor of shape `(1, img_size, img_size)` shifted by `shift_pixels`
    labels[idx]:         scalar binary tensor in {0, 1}

  Concrete Example:
    Suppose img_size=32, shift_pixels=4:
      base_image:    Circle centered at (0, 0)
      shifted_image: Circle centered at (4 pixels to the right, 0)
      label:         0

  Attributes:
    base_images: Tensor of shape `(num_samples, 1, img_size, img_size)`
      with dtype `torch.float32`.
    shifted_images: Tensor of shape `(num_samples, 1, img_size, img_size)`
      with dtype `torch.float32`.
    labels: Tensor of shape `(num_samples,)` with dtype `torch.long` in `{0, 1}`.

  Item Format:
    Indexing `dataset[idx]` returns `(base_image, shifted_image, label)` where:
      - `base_image`: 3D `torch.float32` tensor of shape `(1, img_size, img_size)`.
      - `shifted_image`: 3D `torch.float32` tensor translated by `shift_pixels`.
      - `label`: scalar `torch.long` tensor with shape class identity in `{0, 1}`.
  """

  def __init__(self, num_samples=1000, img_size=32, shift_pixels=4, seed=42):
    """Initializes the spatial invariance dataset.

    Args:
      num_samples: Total number of sample pairs (default: 1000).
      img_size: Image height and width (default: 32).
      shift_pixels: Number of pixels to translate (default: 4).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._img_size = img_size
    self._shift_pixels = shift_pixels

    generator = torch.Generator().manual_seed(seed)
    base_list = []
    shifted_list = []
    labels_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    shift_norm = (2.0 / img_size) * shift_pixels

    for _ in range(num_samples):
      shape_type = torch.randint(0, 2, (1,), generator=generator).item()
      cx = 0.0
      cy = 0.0
      radius = 0.35

      if shape_type == 0:
        base_dist = (x_coords - cx)**2 + (y_coords - cy)**2
        base_mask = (torch.sqrt(base_dist) <= radius).float()
        shifted_dist = (x_coords - (cx + shift_norm))**2 + (y_coords - cy)**2
        shifted_mask = (torch.sqrt(shifted_dist) <= radius).float()
      else:
        base_mask = (((x_coords - cx).abs() <= radius) & ((y_coords - cy).abs() <= radius)).float()
        x_shift_cond = (x_coords - (cx + shift_norm)).abs() <= radius
        y_cond = (y_coords - cy).abs() <= radius
        shifted_mask = (x_shift_cond & y_cond).float()

      base_list.append(base_mask.unsqueeze(0))
      shifted_list.append(shifted_mask.unsqueeze(0))
      labels_list.append(shape_type)

    self._base_images = torch.stack(base_list, dim=0)
    self._shifted_images = torch.stack(shifted_list, dim=0)
    self._labels = torch.tensor(labels_list, dtype=torch.long)

  @property
  def img_size(self):
    """Returns the image resolution."""
    return self._img_size

  @property
  def shift_pixels(self):
    """Returns the pixel translation distance."""
    return self._shift_pixels

  @property
  def base_images(self):
    """Returns the unshifted reference images."""
    return self._base_images

  @property
  def shifted_images(self):
    """Returns the translated image pairs."""
    return self._shifted_images

  @property
  def labels(self):
    """Returns the shape identity labels."""
    return self._labels

  def __getitem__(self, idx):
    """Returns the base image and shifted image pair with class label."""
    return self._base_images[idx], self._shifted_images[idx], self._labels[idx]

  def description(self):
    """Returns a description of the task."""
    return f"Spatial Invariance Pairs ({self._shift_pixels}px, {self._img_size}x{self._img_size})"
