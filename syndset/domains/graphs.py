"""Synthetic graph datasets for evaluating GNN expressivity, shortest path, and SE(3)."""

import math

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset


@register_dataset("graphs/wl_expressivity")
@register_dataset("graphs/csl")
class WLExpressivityDataset(SyntheticDataset):
  """Evaluates GNN expressivity on Circular Skip Link (CSL) graphs.

  CSL graphs are 4-regular non-isomorphic graphs that are indistinguishable by
  standard 1-Weisfeiler-Lehman (1-WL) message passing GNNs. Tests whether higher-order,
  subgraph, or relational GNNs can distinguish isomorphism classes.
  """

  def __init__(
    self,
    num_samples: int = 200,
    num_nodes: int = 16,
    seed: int = 42,
  ):
    """Initializes CSL graph dataset.

    Args:
      num_samples: Number of graph instances to generate (default: 200).
      num_nodes: Number of nodes in the cycle (default: 16).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._num_nodes = num_nodes

    # Skip lengths that yield non-isomorphic 4-regular graphs
    skip_values = [2, 3, 4, 5]
    generator = torch.Generator().manual_seed(seed)

    self._adj_list = []
    self._node_feats_list = []
    self._labels_list = []

    for _ in range(num_samples):
      class_idx = torch.randint(0, len(skip_values), (1,), generator=generator).item()
      skip = skip_values[class_idx]

      adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
      for i in range(num_nodes):
        # Cycle neighbors
        adj[i, (i + 1) % num_nodes] = 1.0
        adj[i, (i - 1 + num_nodes) % num_nodes] = 1.0
        # Skip neighbors
        adj[i, (i + skip) % num_nodes] = 1.0
        adj[i, (i - skip + num_nodes) % num_nodes] = 1.0

      # Permute node ordering to prevent position triviality
      perm = torch.randperm(num_nodes, generator=generator)
      adj = adj[perm][:, perm]

      # Uninformative uniform node features (forces relying purely on topology)
      node_feats = torch.ones((num_nodes, 1), dtype=torch.float32)

      self._adj_list.append(adj)
      self._node_feats_list.append(node_feats)
      self._labels_list.append(class_idx)

    self._adj = torch.stack(self._adj_list, dim=0)
    self._node_feats = torch.stack(self._node_feats_list, dim=0)
    self._labels = torch.tensor(self._labels_list, dtype=torch.long)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing adjacency matrix, node features, and skip class."""
    return DataSample(
      adjacency=self._adj[idx],
      node_features=self._node_feats[idx],
      label=self._labels[idx],
    )

  def description(self) -> str:
    """Returns a description of the CSL expressivity dataset."""
    return f"CSL Graphs (1-WL limit, {self._num_nodes} nodes, 4 classes)"


@register_dataset("graphs/shortest_path")
class ShortestPathDataset(SyntheticDataset):
  """Procedurally generates graphs with ground-truth Bellman-Ford shortest paths.

  Tests neural algorithmic reasoning and sequential message-passing horizons.
  """

  def __init__(
    self,
    num_samples: int = 500,
    num_nodes: int = 10,
    edge_prob: float = 0.35,
    seed: int = 42,
  ):
    """Initializes shortest path graph dataset.

    Args:
      num_samples: Number of graph instances (default: 500).
      num_nodes: Number of nodes per graph (default: 10).
      edge_prob: Probability of edge existence (default: 0.35).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._num_nodes = num_nodes

    generator = torch.Generator().manual_seed(seed)
    self._adj_list = []
    self._src_dst_list = []
    self._dist_list = []

    for _ in range(num_samples):
      # Random connected-ish Erdős-Rényi graph with positive edge weights
      weights = torch.rand(num_nodes, num_nodes, generator=generator) * 5.0 + 1.0
      mask = (torch.rand(num_nodes, num_nodes, generator=generator) < edge_prob).float()
      # Zero diagonal, make symmetric
      mask.fill_diagonal_(0.0)
      adj = torch.triu(mask * weights, diagonal=1)
      adj = adj + adj.T

      # Compute shortest paths using Floyd-Warshall
      dist_mat = torch.full((num_nodes, num_nodes), float("inf"))
      dist_mat.fill_diagonal_(0.0)
      has_edge = adj > 0.0
      dist_mat[has_edge] = adj[has_edge]

      for k in range(num_nodes):
        dist_mat = torch.minimum(dist_mat, dist_mat[:, k : k + 1] + dist_mat[k : k + 1, :])

      src = torch.randint(0, num_nodes, (1,), generator=generator).item()
      dst = (src + torch.randint(1, num_nodes, (1,), generator=generator).item()) % num_nodes

      d_val = dist_mat[src, dst]
      if math.isinf(d_val.item()):
        d_val = torch.tensor(99.0, dtype=torch.float32)

      self._adj_list.append(adj)
      self._src_dst_list.append(torch.tensor([src, dst], dtype=torch.long))
      self._dist_list.append(d_val)

    self._adj = torch.stack(self._adj_list, dim=0)
    self._src_dst = torch.stack(self._src_dst_list, dim=0)
    self._dist = torch.stack(self._dist_list, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing adjacency, (src, dst) pair, and shortest distance."""
    return DataSample(
      adjacency=self._adj[idx],
      source_dest=self._src_dst[idx],
      distance=self._dist[idx],
    )

  def description(self) -> str:
    """Returns a description of the shortest path dataset."""
    return f"Shortest Path Graphs ({self._num_nodes} nodes, p={self._num_nodes})"


@register_dataset("graphs/se3_equivariance")
@register_dataset("graphs/molecular_invariance")
class SE3EquivarianceDataset(SyntheticDataset):
  """Generates 3D geometric point clouds with exact SE(3) invariant electrostatic targets.

  Computes analytical pairwise Coulomb energy y = sum_{i < j} q_i q_j / ||r_i - r_j||,
  which is rigorously invariant under 3D rigid rotations and translations in SE(3).
  """

  def __init__(
    self,
    num_samples: int = 500,
    num_atoms: int = 8,
    seed: int = 42,
  ):
    """Initializes 3D molecular point cloud dataset.

    Args:
      num_samples: Number of molecular geometries (default: 500).
      num_atoms: Number of atoms per molecule (default: 8).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._num_atoms = num_atoms

    generator = torch.Generator().manual_seed(seed)
    coords_list = []
    charges_list = []
    energy_list = []

    for _ in range(num_samples):
      # Atom 3D positions in [-2.0, 2.0]
      coords = torch.randn(num_atoms, 3, generator=generator) * 1.5
      charges = torch.randint(1, 6, (num_atoms,), generator=generator).float()

      # Pairwise distances
      diffs = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
      dist = torch.norm(diffs, p=2, dim=-1) + 1e-4  # (N, N)

      q_matrix = charges.unsqueeze(0) * charges.unsqueeze(1)
      coulomb = q_matrix / dist
      triu_indices = torch.triu_indices(num_atoms, num_atoms, offset=1)
      total_energy = torch.sum(coulomb[triu_indices[0], triu_indices[1]])

      coords_list.append(coords)
      charges_list.append(charges)
      energy_list.append(total_energy)

    self._coords = torch.stack(coords_list, dim=0)
    self._charges = torch.stack(charges_list, dim=0)
    self._energy = torch.stack(energy_list, dim=0)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing 3D atomic coordinates, charges, and invariant energy."""
    return DataSample(
      coordinates=self._coords[idx],
      charges=self._charges[idx],
      energy=self._energy[idx],
    )

  def description(self) -> str:
    """Returns a description of the 3D equivariance dataset."""
    return f"3D Geometric Point Clouds ({self._num_atoms} atoms, SE(3) Coulomb energy)"
