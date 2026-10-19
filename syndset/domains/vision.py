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
      seed: Random seed (default: 42).
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


class TextureVsShapeDataset(SyntheticDataset):
  """Tests whether a vision model relies primarily on shape contours or surface textures.

  Produces samples where a primary shape (circle vs square) is overlaid with
  a conflicting high-frequency texture (horizontal stripes vs vertical stripes).
  Allows checking whether an architecture exhibits shape bias (typical of humans
  and ViTs) or texture bias (typical of standard CNNs).
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
