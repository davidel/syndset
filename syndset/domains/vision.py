"""Synthetic vision datasets for testing spatial inductive biases and convolutions."""

import math

import torch

from syndset.domains.base import SyntheticDataset


class SyntheticShapesDataset(SyntheticDataset):
  """Procedurally generates geometric shapes (circles, squares, triangles, pluses).

  Provides a clean, controllable testbed for vision architectures (CNNs, ViTs,
  MLP-Mixers) to test spatial pattern recognition, translation invariance,
  and background noise robustness without needing heavy external image datasets.
  """

  def __init__(self, num_samples=1000, img_size=32, channels=1, noise_std=0.1, seed=42):
    """Initializes the synthetic shapes dataset.

    Args:
      num_samples: Number of images to generate (default: 1000).
      img_size: Height and width in pixels (default: 32).
      channels: Number of color channels (default: 1).
      noise_std: Standard deviation of background Gaussian noise (default: 0.1).
      seed: Random seed (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self.img_size = img_size
    self.channels = channels
    self.noise_std = noise_std
    self.num_classes = 4

    generator = torch.Generator().manual_seed(seed)
    images_list = []
    labels_list = []

    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    for _ in range(num_samples):
      label = torch.randint(0, 4, (1,), generator=generator).item()
      cx = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
      cy = (torch.rand(1, generator=generator).item() - 0.5) * 0.6
      radius = 0.2 + torch.rand(1, generator=generator).item() * 0.2

      dx = x_coords - cx
      dy = y_coords - cy
      dist = torch.sqrt(dx**2 + dy**2)

      mask = torch.zeros((img_size, img_size), dtype=torch.float32)

      if label == 0:
        # Circle
        mask = (dist <= radius).float()
      elif label == 1:
        # Square
        mask = ((dx.abs() <= radius) & (dy.abs() <= radius)).float()
      elif label == 2:
        # Triangle (pointing upwards)
        in_height = (dy >= -radius) & (dy <= radius)
        in_width = dx.abs() <= (radius - dy * 0.5)
        mask = (in_height & in_width).float()
      else:
        # Plus / Cross
        arm_w = radius * 0.35
        horiz = (dy.abs() <= arm_w) & (dx.abs() <= radius)
        vert = (dx.abs() <= arm_w) & (dy.abs() <= radius)
        mask = (horiz | vert).float()

      # Add noise and replicate across channels
      noise = torch.randn((img_size, img_size), generator=generator) * noise_std
      img = (mask + noise).clamp(0.0, 1.0)
      img = img.unsqueeze(0).repeat(channels, 1, 1)

      images_list.append(img)
      labels_list.append(label)

    self._images = torch.stack(images_list, dim=0)
    self._labels = torch.tensor(labels_list, dtype=torch.long)

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
    dims_str = f"{self.channels}x{self.img_size}x{self.img_size}"
    return f"Synthetic Shapes ({self.num_classes} classes, {dims_str})"


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
    self.img_size = img_size

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
    return f"Texture vs Shape Conflict ({self.img_size}x{self.img_size})"


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
    self.img_size = img_size
    self.shift_pixels = shift_pixels

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
        base_dist = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
        base_mask = (torch.sqrt(base_dist) <= radius).float()
        shifted_dist = (x_coords - (cx + shift_norm)) ** 2 + (y_coords - cy) ** 2
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
    return f"Spatial Invariance Pairs ({self.shift_pixels}px, {self.img_size}x{self.img_size})"
