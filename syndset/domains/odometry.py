"""Synthetic visual odometry, camera ego-motion, and optical flow datasets."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> torch.Tensor:
  """Computes a 3x3 rotation matrix from Euler angles (in radians)."""
  cr, sr = math.cos(roll), math.sin(roll)
  cp, sp = math.cos(pitch), math.sin(pitch)
  cy, sy = math.cos(yaw), math.sin(yaw)

  r_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=torch.float32)
  r_y = torch.tensor([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=torch.float32)
  r_z = torch.tensor([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

  return r_z @ r_y @ r_x


@register_dataset("odometry/visual_odometry")
@register_dataset("odometry/vo")
class VisualOdometryDataset(SyntheticDataset):
  """Simulates pinhole camera projection and SE(3) ego-motion for visual odometry.

  Generates 3D landmark points projected onto consecutive virtual camera frames
  under smooth rigid body motion.

  Item Format (DataSample):
    - `points_prev`: 2D pixel coordinates at step t-1 (shape: `(num_points, 2)`).
    - `points_curr`: 2D pixel coordinates at step t (shape: `(num_points, 2)`).
    - `rel_rotation`: Relative 3x3 rotation matrix Delta R in SO(3).
    - `rel_translation`: Relative 3D translation vector Delta t in R^3.
    - `depth_curr`: Depth (Z-coordinate) of landmarks in the current camera frame.
  """

  def __init__(
    self,
    num_samples: int = 500,
    num_points: int = 50,
    img_size: int = 128,
    focal_length: float = 100.0,
    noise_std: float = 0.01,
    seed: int = 42,
  ):
    """Initializes visual odometry dataset.

    Args:
      num_samples: Number of frame pairs to simulate (default: 500).
      num_points: Number of 3D landmark feature points (default: 50).
      img_size: Virtual image resolution in pixels (default: 128).
      focal_length: Pinhole camera focal length fx = fy (default: 100.0).
      noise_std: Measurement noise added to 2D projected pixel locations (default: 0.01).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._num_points = num_points
    self._img_size = img_size
    self._focal_length = focal_length
    self._noise_std = noise_std

    generator = torch.Generator().manual_seed(seed)
    cx = img_size / 2.0
    cy = img_size / 2.0

    self._points_prev_list = []
    self._points_curr_list = []
    self._rel_rot_list = []
    self._rel_trans_list = []
    self._depth_list = []

    for _ in range(num_samples):
      # 1. Generate 3D landmark points in front of the camera (Z in [1.5, 4.0])
      pts_x = (torch.rand(num_points, 1, generator=generator) - 0.5) * 2.0
      pts_y = (torch.rand(num_points, 1, generator=generator) - 0.5) * 2.0
      pts_z = torch.rand(num_points, 1, generator=generator) * 2.5 + 1.5
      pts_w = torch.cat([pts_x, pts_y, pts_z], dim=1)  # (N, 3)

      # Small random SE(3) camera motion (roll, pitch, yaw in [-0.08, 0.08] rad)
      roll = (torch.rand(1, generator=generator).item() - 0.5) * 0.16
      pitch = (torch.rand(1, generator=generator).item() - 0.5) * 0.16
      yaw = (torch.rand(1, generator=generator).item() - 0.5) * 0.16
      r_1 = _euler_to_rotation_matrix(roll, pitch, yaw)

      dx = (torch.rand(1, generator=generator).item() - 0.5) * 0.15
      dy = (torch.rand(1, generator=generator).item() - 0.5) * 0.15
      dz = torch.rand(1, generator=generator).item() * 0.15 + 0.05  # forward motion
      t_1 = torch.tensor([dx, dy, dz], dtype=torch.float32)

      # Project into frame 0: P_c0 = R_0 @ P_w + t_0
      p_c0 = pts_w
      u0 = focal_length * (p_c0[:, 0] / p_c0[:, 2]) + cx
      v0 = focal_length * (p_c0[:, 1] / p_c0[:, 2]) + cy
      noise_0 = torch.randn(num_points, 2, generator=generator) * noise_std
      pts_2d_0 = torch.stack([u0, v0], dim=1) + noise_0

      # Project into frame 1: P_c1 = R_1 @ (P_w - t_1)
      p_c1 = (pts_w - t_1.unsqueeze(0)) @ r_1.T
      u1 = focal_length * (p_c1[:, 0] / p_c1[:, 2]) + cx
      v1 = focal_length * (p_c1[:, 1] / p_c1[:, 2]) + cy
      noise_1 = torch.randn(num_points, 2, generator=generator) * noise_std
      pts_2d_1 = torch.stack([u1, v1], dim=1) + noise_1

      # Relative motion Delta R = R_1, Delta t = t_1
      self._points_prev_list.append(pts_2d_0)
      self._points_curr_list.append(pts_2d_1)
      self._rel_rot_list.append(r_1)
      self._rel_trans_list.append(t_1)
      self._depth_list.append(p_c1[:, 2])

    self._points_prev = torch.stack(self._points_prev_list, dim=0)
    self._points_curr = torch.stack(self._points_curr_list, dim=0)
    self._rel_rot = torch.stack(self._rel_rot_list, dim=0)
    self._rel_trans = torch.stack(self._rel_trans_list, dim=0)
    self._depth = torch.stack(self._depth_list, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing 2D landmark pairs and SE(3) ground truth."""
    return DataSample(
      points_prev=self._points_prev[idx],
      points_curr=self._points_curr[idx],
      rel_rotation=self._rel_rot[idx],
      rel_translation=self._rel_trans[idx],
      depth=self._depth[idx],
    )

  def description(self) -> str:
    """Returns a description of the dataset."""
    desc = f"{self._num_points} landmarks, {self._img_size}x{self._img_size}"
    return f"Visual Odometry SE(3) ({desc})"


@register_dataset("odometry/optical_flow")
class OpticalFlowDataset(SyntheticDataset):
  """Generates paired 2D images and analytical dense optical flow vector fields.

  Computes exact analytical per-pixel motion vectors (u, v) in R^(2 x H x W)
  induced by known rigid 2D planar transformations.
  """

  def __init__(
    self,
    num_samples: int = 500,
    img_size: int = 32,
    max_displacement: float = 3.0,
    seed: int = 42,
  ):
    """Initializes optical flow dataset.

    Args:
      num_samples: Number of image pairs to generate (default: 500).
      img_size: Image height and width in pixels (default: 32).
      max_displacement: Maximum pixel displacement per axis (default: 3.0).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._img_size = img_size
    self._max_displacement = max_displacement

    generator = torch.Generator().manual_seed(seed)
    y_grid, x_grid = torch.meshgrid(
      torch.arange(img_size, dtype=torch.float32),
      torch.arange(img_size, dtype=torch.float32),
      indexing="ij",
    )

    self._img_prev_list = []
    self._img_curr_list = []
    self._flow_list = []

    for _ in range(num_samples):
      # Discrete motion: translation (tx, ty)
      tx = (torch.rand(1, generator=generator).item() - 0.5) * 2.0 * max_displacement
      ty = (torch.rand(1, generator=generator).item() - 0.5) * 2.0 * max_displacement

      # Base image: smooth 2D Gaussian blob
      blob_cx = img_size / 2.0 + (torch.rand(1, generator=generator).item() - 0.5) * 8.0
      blob_cy = img_size / 2.0 + (torch.rand(1, generator=generator).item() - 0.5) * 8.0
      sigma = 4.0

      dist_0 = ((x_grid - blob_cx)**2 + (y_grid - blob_cy)**2) / (2.0 * sigma**2)
      img_0 = torch.exp(-dist_0).unsqueeze(0)

      # Shifted image at step 1
      dist_1 = ((x_grid - (blob_cx + tx))**2 + (y_grid - (blob_cy + ty))**2) / (2.0 * sigma**2)
      img_1 = torch.exp(-dist_1).unsqueeze(0)

      # Analytical flow field: displacement at every pixel is (tx, ty)
      flow_u = torch.full((img_size, img_size), tx, dtype=torch.float32)
      flow_v = torch.full((img_size, img_size), ty, dtype=torch.float32)
      flow = torch.stack([flow_u, flow_v], dim=0)  # (2, H, W)

      self._img_prev_list.append(img_0)
      self._img_curr_list.append(img_1)
      self._flow_list.append(flow)

    self._img_prev = torch.stack(self._img_prev_list, dim=0)
    self._img_curr = torch.stack(self._img_curr_list, dim=0)
    self._flow = torch.stack(self._flow_list, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing frame pairs and analytical 2D flow field."""
    return DataSample(
      image_prev=self._img_prev[idx],
      image_curr=self._img_curr[idx],
      flow=self._flow[idx],
    )

  def description(self) -> str:
    """Returns a description of the optical flow task."""
    return f"Dense Optical Flow (2x{self._img_size}x{self._img_size})"
