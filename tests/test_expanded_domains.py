"""Tests for newly expanded synthetic dataset domains."""

import pytest
import torch

import syndset as syn
from syndset.domains.base import DataSample


def test_data_sample_container():
  """Tests DataSample tuple unpacking, dict access, and attribute access."""
  t1 = torch.tensor([1.0, 2.0])
  t2 = torch.tensor([3.0, 4.0])
  sample = DataSample(features=t1, label=t2)

  # Tuple unpacking
  unpacked_1, unpacked_2 = sample
  assert torch.equal(unpacked_1, t1)
  assert torch.equal(unpacked_2, t2)

  # Dict-style access
  assert torch.equal(sample["features"], t1)
  assert torch.equal(sample["label"], t2)

  # Attribute-style access
  assert torch.equal(sample.features, t1)
  assert torch.equal(sample.label, t2)

  # Keys, values, and dict conversion
  assert "features" in sample.keys()
  assert "label" in sample.keys()
  d = sample.to_dict()
  assert isinstance(d, dict)
  assert "features" in d


def test_dataset_registry_and_factory():
  """Tests dataset registration, listing, and factory loading."""
  all_names = syn.list_datasets()
  assert len(all_names) > 30
  assert "multimodal/shapes" in all_names
  assert "odometry/vo" in all_names
  assert "rl/cartpole" in all_names

  # Factory load with kwargs
  ds = syn.load_dataset("multimodal/shapes", num_samples=10, img_size=16)
  assert len(ds) == 10
  assert ds[0].image.shape == (3, 16, 16)

  with pytest.raises(KeyError):
    syn.load_dataset("non_existent_domain/unknown_dataset")


def test_dataloader_integration():
  """Tests wrapping synthetic datasets with PyTorch DataLoader."""
  ds = syn.load_dataset("generative/score", num_samples=20, dim=3)
  loader = ds.to_dataloader(batch_size=5, shuffle=True)
  assert len(loader) == 4

  for batch in loader:
    assert len(batch) == 2
    # Point and score batched
    assert batch[0].shape == (5, 3)
    assert batch[1].shape == (5, 3)
    break


def test_multimodal_shapes_dataset():
  """Tests multimodal paired image-caption and negative captions."""
  ds = syn.MultimodalShapesDataset(num_samples=15, img_size=24, num_objects=2, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.image.shape == (3, 24, 24)
  assert sample.input_ids.shape[0] == 64
  assert isinstance(sample.caption, str)
  assert len(sample.caption) > 0
  assert isinstance(sample.attr_negative, str)
  assert isinstance(sample.pos_negative, str)
  assert sample.caption != sample.attr_negative


def test_multimodal_vqa_dataset():
  """Tests synthetic VQA dataset."""
  ds = syn.VisualQuestionAnsweringDataset(num_samples=10, img_size=24, seed=42)
  assert len(ds) == 10
  sample = ds[0]
  assert sample.image.shape == (3, 24, 24)
  assert isinstance(sample.question, str)
  assert isinstance(sample.answer, str)
  assert sample.answer_id.dtype == torch.long


def test_visual_odometry_dataset():
  """Tests visual odometry pinhole projection and SE(3) ego-motion."""
  ds = syn.VisualOdometryDataset(num_samples=10, num_points=30, img_size=64, seed=42)
  assert len(ds) == 10
  sample = ds[0]
  assert sample.points_prev.shape == (30, 2)
  assert sample.points_curr.shape == (30, 2)
  assert sample.rel_rotation.shape == (3, 3)
  assert sample.rel_translation.shape == (3,)
  assert sample.depth.shape == (30,)

  # Check SO(3) orthogonality: R @ R.T ~= I
  r = sample.rel_rotation
  eye = torch.eye(3)
  assert torch.allclose(r @ r.T, eye, atol=1e-5)


def test_optical_flow_dataset():
  """Tests dense 2D optical flow generation."""
  ds = syn.OpticalFlowDataset(num_samples=10, img_size=24, seed=42)
  assert len(ds) == 10
  sample = ds[0]
  assert sample.image_prev.shape == (1, 24, 24)
  assert sample.image_curr.shape == (1, 24, 24)
  assert sample.flow.shape == (2, 24, 24)


def test_rl_cartpole_mdp():
  """Tests CartPole MDP analytical simulation."""
  env = syn.CartPoleMDP(seed=42)
  s0 = env.reset()
  assert s0.shape == (4,)

  s1, r, done, _ = env.step(1)
  assert s1.shape == (4,)
  assert isinstance(r, float)
  assert isinstance(done, bool)


def test_rl_cartpole_offline_dataset():
  """Tests CartPole offline trajectory buffer under different policies."""
  for policy in ["expert", "suboptimal", "random"]:
    ds = syn.CartPoleOfflineDataset(num_samples=30, policy=policy, seed=42)
    assert len(ds) == 30
    sample = ds[0]
    assert sample.state.shape == (4,)
    assert sample.action.dtype == torch.long
    assert sample.next_state.shape == (4,)
    assert sample.done.dtype == torch.float32


def test_rl_gridworld_dataset():
  """Tests Gridworld discrete navigation dataset."""
  ds = syn.GridworldDataset(num_samples=25, grid_size=4, seed=42)
  assert len(ds) == 25
  sample = ds[0]
  assert sample.state.shape == (2,)
  assert sample.action.item() in [0, 1, 2, 3]
  assert sample.next_state.shape == (2,)


def test_audio_fm_chirp_dataset():
  """Tests 1D FM chirp audio waveform generation."""
  ds = syn.FMChirpDataset(num_samples=15, seq_len=128, sample_rate=500.0, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.waveform.shape == (1, 128)
  assert sample.initial_freq > 0.0
  assert sample.chirp_rate > 0.0


def test_audio_sound_events_dataset():
  """Tests polyphonic sound event temporal intervals."""
  ds = syn.PolyphonicSoundEventDataset(num_samples=10, seq_len=128, num_classes=3, seed=42)
  assert len(ds) == 10
  sample = ds[0]
  assert sample.waveform.shape == (1, 128)
  assert sample.event_mask.shape == (3, 128)


def test_generative_analytical_score():
  """Tests exact analytical score vector field on Gaussian Mixture."""
  ds = syn.AnalyticalScoreDataset(num_samples=20, dim=2, num_components=3, seed=42)
  assert len(ds) == 20
  sample = ds[0]
  assert sample.point.shape == (2,)
  assert sample.score.shape == (2,)
  assert not torch.isnan(sample.score).any()


def test_generative_flow_matching():
  """Tests optimal transport flow matching velocity targets."""
  ds = syn.FlowMatchingDataset(num_samples=20, dim=2, seed=42)
  assert len(ds) == 20
  sample = ds[0]
  assert sample.x_t.shape == (2,)
  assert sample.time.shape == (1,)
  assert sample.target_velocity.shape == (2,)


def test_generative_toy_manifolds():
  """Tests 2D toy manifold distributions."""
  for m_type in ["swiss_roll", "pinwheel", "checkerboard"]:
    ds = syn.ToyManifold2DDataset(num_samples=30, manifold_type=m_type, seed=42)
    assert len(ds) == 30
    sample = ds[0]
    assert sample.point.shape == (2,)


def test_graphs_wl_expressivity():
  """Tests Circular Skip Link (CSL) 1-WL expressivity benchmark."""
  ds = syn.WLExpressivityDataset(num_samples=16, num_nodes=12, seed=42)
  assert len(ds) == 16
  sample = ds[0]
  assert sample.adjacency.shape == (12, 12)
  assert sample.node_features.shape == (12, 1)
  assert sample.label.dtype == torch.long


def test_graphs_shortest_path():
  """Tests shortest path graph reasoning."""
  ds = syn.ShortestPathDataset(num_samples=15, num_nodes=8, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.adjacency.shape == (8, 8)
  assert sample.source_dest.shape == (2,)
  assert sample.distance.shape == ()


def test_graphs_se3_equivariance():
  """Tests 3D molecular point clouds with exact SE(3) Coulomb energy."""
  ds = syn.SE3EquivarianceDataset(num_samples=15, num_atoms=6, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.coordinates.shape == (6, 3)
  assert sample.charges.shape == (6,)
  assert sample.energy.shape == ()
  assert sample.energy > 0.0


def test_neural_fields_sdf():
  """Tests analytical signed distance functions and surface normals."""
  for shape in ["sphere", "box", "torus"]:
    ds = syn.AnalyticalSDFDataset(num_samples=25, shape_type=shape, seed=42)
    assert len(ds) == 25
    sample = ds[0]
    assert sample.point.shape == (3,)
    assert sample.distance.shape == (1,)
    assert sample.normal.shape == (3,)
    # Normals should be unit vectors
    norm = torch.norm(sample.normal, p=2)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-3)


def test_neural_fields_toy_nerf():
  """Tests toy NeRF ray marching colored sphere dataset."""
  ds = syn.ToyNeRFDataset(num_samples=20, radius=0.6, seed=42)
  assert len(ds) == 20
  sample = ds[0]
  assert sample.ray_origin.shape == (3,)
  assert sample.ray_dir.shape == (3,)
  assert sample.color.shape == (3,)
  assert (sample.color >= 0.0).all() and (sample.color <= 1.0).all()


def test_interpretability_modular_arithmetic():
  """Tests modular arithmetic grokking dataset."""
  ds = syn.ModularArithmeticDataset(prime=13, operation="add", seed=42)
  assert len(ds) == 13 * 13
  sample = ds[0]
  assert sample.inputs.shape == (2,)
  assert sample.target.dtype == torch.long
  assert sample.target < 13


def test_interpretability_feature_superposition():
  """Tests feature superposition dataset for Sparse Autoencoders."""
  ds = syn.FeatureSuperpositionDataset(num_samples=30, num_features=32, bottleneck_dim=8, seed=42)
  assert len(ds) == 30
  sample = ds[0]
  assert sample.activation.shape == (8,)
  assert sample.true_features.shape == (32,)
  assert ds.dictionary.shape == (8, 32)


def test_physics_heat_equation():
  """Tests 1D Heat equation Fourier analytical solution."""
  ds = syn.HeatEquationDataset(num_samples=30, alpha=0.05, num_modes=3, seed=42)
  assert len(ds) == 30
  sample = ds[0]
  assert sample.coordinates.shape == (2,)
  assert sample.solution.shape == (1,)
  assert not torch.isnan(sample.solution).any()


def test_physics_burgers_equation():
  """Tests 1D Burgers equation trajectory simulation."""
  ds = syn.BurgersEquationDataset(num_samples=15, spatial_points=32, nu=0.02, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.initial_state.shape == (1, 32)
  assert sample.future_state.shape == (1, 32)


def test_continual_rotating_stream():
  """Tests rotating manifold continual learning stream."""
  ds = syn.RotatingManifoldStreamDataset(num_samples_per_task=20, num_tasks=3, seed=42)
  assert len(ds) == 60
  sample = ds[0]
  assert sample.features.shape == (2,)
  assert sample.label.dtype == torch.long
  assert sample.task_id.dtype == torch.long


def test_continual_spurious_correlation():
  """Tests spurious correlation shortcut learning dataset."""
  ds_train = syn.SpuriousCorrelationDataset(num_samples=25, split="train", seed=42)
  ds_test = syn.SpuriousCorrelationDataset(num_samples=25, split="test", seed=42)
  assert len(ds_train) == 25
  assert len(ds_test) == 25
  sample = ds_train[0]
  assert sample.features.shape == (2,)
  assert sample.label.dtype == torch.long


def test_llm_needle_in_haystack():
  """Tests long-context Needle In Haystack dataset."""
  ds = syn.NeedleInHaystackDataset(num_samples=15,
                                   seq_len=64,
                                   vocab_size=32,
                                   needle_depth=0.5,
                                   seed=42)
  assert len(ds) == 15
  assert ds.seq_len == 64
  assert ds.needle_depth == 0.5
  sample = ds[0]
  assert sample.inputs.shape[0] == 64
  assert sample.target.dtype == torch.long


def test_llm_structured_grammar():
  """Tests structured JSON grammar dataset."""
  ds = syn.StructuredGrammarDataset(num_samples=15, seq_len=32, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.inputs.shape[0] == 31
  assert sample.target.shape[0] == 31


def test_llm_reasoning_dag():
  """Tests multi-hop transitive DAG reasoning."""
  ds = syn.ReasoningDAGDataset(num_samples=15, num_hops=3, vocab_size=32, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  # 3 hops -> 6 facts + 1 query = 7 tokens
  assert sample.inputs.shape[0] == 7
  assert sample.target.dtype == torch.long


def test_vision_occluded_detection():
  """Tests depth-ordered occluded object detection."""
  ds = syn.OccludedDetectionDataset(num_samples=15, img_size=24, num_objects=3, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.image.shape == (1, 24, 24)
  assert sample.boxes.shape == (3, 4)
  assert sample.labels.shape == (3,)
  assert sample.occlusion_rates.shape == (3,)


def test_vision_articulated_keypoints():
  """Tests 2D articulated stick figure keypoint dataset."""
  ds = syn.ArticulatedKeypointsDataset(num_samples=15, img_size=24, seed=42)
  assert len(ds) == 15
  sample = ds[0]
  assert sample.image.shape == (1, 24, 24)
  assert sample.keypoints.shape == (7, 2)
  # Normalized coordinates in [0, 1]
  assert (sample.keypoints >= 0.0).all() and (sample.keypoints <= 1.0).all()
