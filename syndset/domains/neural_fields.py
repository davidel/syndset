"""Synthetic neural fields, signed distance functions (SDFs), and NeRF volume rendering datasets."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


@register_dataset("neural_fields/analytical_sdf")
@register_dataset("neural_fields/sdf")
class AnalyticalSDFDataset(SyntheticDataset):
  """Evaluates coordinate-based neural implicit representations on exact analytical SDFs.

  Computes closed-form signed distance d(x, y, z) and analytical surface normal
  vectors n = nabla d for primitive 3D geometries (sphere, box, torus).
  Tests Eikonal loss ||nabla f(x)|| = 1 and zero-level set reconstruction.
  """

  def __init__(
    self,
    num_samples: int = 1000,
    shape_type: str = "sphere",
    seed: int = 42,
  ):
    """Initializes analytical SDF dataset.

    Args:
      num_samples: Number of 3D query points to evaluate (default: 1000).
      shape_type: Primitive solid geometry ('sphere', 'box', 'torus').
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._shape_type = shape_type

    generator = torch.Generator().manual_seed(seed)
    # Sample query points in bounding cube [-1.5, 1.5]^3
    points = (torch.rand(num_samples, 3, generator=generator) - 0.5) * 3.0

    if shape_type == "sphere":
      radius = 0.8
      dist = torch.norm(points, p=2, dim=-1)
      sdf = dist - radius
      normals = points / (dist.unsqueeze(-1) + 1e-12)
    elif shape_type == "box":
      b = torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32)
      q = points.abs() - b.unsqueeze(0)
      outside = torch.norm(torch.clamp(q, min=0.0), p=2, dim=-1)
      inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
      sdf = outside + inside
      normals = torch.sign(points) * (q == torch.max(q, dim=-1, keepdim=True).values).float()
      normals = normals / (torch.norm(normals, p=2, dim=-1, keepdim=True) + 1e-12)
    else:  # Torus: R=0.7, r=0.25
      R = 0.7
      r = 0.25
      x, y, z = points[:, 0], points[:, 1], points[:, 2]
      q_xy = torch.sqrt(x**2 + y**2) - R
      sdf = torch.sqrt(q_xy**2 + z**2) - r
      grad_x = x * q_xy / (torch.sqrt(x**2 + y**2) + 1e-12)
      grad_y = y * q_xy / (torch.sqrt(x**2 + y**2) + 1e-12)
      grad_z = z
      grad = torch.stack([grad_x, grad_y, grad_z], dim=-1)
      normals = grad / (torch.norm(grad, p=2, dim=-1, keepdim=True) + 1e-12)

    self._points = points
    self._sdf = sdf.unsqueeze(-1)  # (N, 1)
    self._normals = normals  # (N, 3)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing 3D query point, true signed distance, and surface normal."""
    return DataSample(
      point=self._points[idx],
      distance=self._sdf[idx],
      normal=self._normals[idx],
    )

  def description(self) -> str:
    """Returns a description of the SDF dataset."""
    return f"Analytical SDF (shape='{self._shape_type}', 3D points)"


@register_dataset("neural_fields/toy_nerf")
@register_dataset("neural_fields/nerf")
class ToyNeRFDataset(SyntheticDataset):
  """Simulates ray casting and analytical volume rendering of a colored 3D sphere.

  Computes exact ray-sphere intersections for camera rays and evaluates
  emitted RGB color for testing NeRF ray marchers and coordinate MLPs.
  """

  def __init__(
    self,
    num_samples: int = 500,
    radius: float = 0.5,
    seed: int = 42,
  ):
    """Initializes toy NeRF ray dataset.

    Args:
      num_samples: Number of camera rays to evaluate (default: 500).
      radius: Radius of the colored virtual sphere (default: 0.5).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._radius = radius

    generator = torch.Generator().manual_seed(seed)
    origins_list = []
    dirs_list = []
    rgb_list = []

    # Camera orbiting on a sphere of radius 2.0 looking at the origin (0, 0, 0)
    for _ in range(num_samples):
      phi = torch.rand(1, generator=generator).item() * 2.0 * math.pi
      theta = torch.rand(1, generator=generator).item() * math.pi * 0.6 + 0.2
      cam_x = 2.0 * math.sin(theta) * math.cos(phi)
      cam_y = 2.0 * math.sin(theta) * math.sin(phi)
      cam_z = 2.0 * math.cos(theta)
      cam_origin = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float32)

      # Target point near origin
      target = (torch.rand(3, generator=generator) - 0.5) * 1.0
      ray_dir = target - cam_origin
      ray_dir = ray_dir / torch.norm(ray_dir, p=2)

      # Analytical ray-sphere intersection: ||o + t*d||^2 = R^2
      # t^2 + 2*(o.d)*t + (||o||^2 - R^2) = 0
      b = 2.0 * torch.dot(cam_origin, ray_dir).item()
      c = torch.dot(cam_origin, cam_origin).item() - radius**2
      discriminant = b**2 - 4.0 * c

      if discriminant > 0:
        # Intersects sphere: color depends on intersection normal
        t_hit = (-b - math.sqrt(discriminant)) / 2.0
        hit_pt = cam_origin + t_hit * ray_dir
        normal = hit_pt / radius
        # Map normal [-1, 1] to RGB [0, 1]
        rgb = (normal + 1.0) * 0.5
      else:
        # Background: dark blue
        rgb = torch.tensor([0.05, 0.05, 0.2], dtype=torch.float32)

      origins_list.append(cam_origin)
      dirs_list.append(ray_dir)
      rgb_list.append(rgb)

    self._origins = torch.stack(origins_list, dim=0)
    self._dirs = torch.stack(dirs_list, dim=0)
    self._rgb = torch.stack(rgb_list, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing ray origin, ray direction, and ground-truth RGB."""
    return DataSample(
      ray_origin=self._origins[idx],
      ray_dir=self._dirs[idx],
      color=self._rgb[idx],
    )

  def description(self) -> str:
    """Returns a description of the toy NeRF dataset."""
    return f"Toy NeRF Ray Marching (colored sphere R={self._radius})"
