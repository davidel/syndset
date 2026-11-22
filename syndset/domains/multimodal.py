"""Synthetic multimodal and vision-language datasets for CLIP and VLM architectures."""

import torch

from syndset.domains.base import DataSample, SyntheticDataset, register_dataset
from syndset.domains.vision import rasterize_shape

COLOR_NAMES = ["red", "green", "blue", "yellow", "magenta", "cyan"]
COLOR_RGB = [
  (1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, 0.0, 1.0),
  (1.0, 1.0, 0.0),
  (1.0, 0.0, 1.0),
  (0.0, 1.0, 1.0),
]

SHAPE_NAMES_MM = ["circle", "square", "triangle", "cross"]
QUADRANT_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]
QUADRANT_COORDS = [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]


def _tokenize_text(text: str, max_len: int = 32) -> torch.Tensor:
  """Simple byte/character tokenization for zero-dependency textual representation."""
  tokens = [ord(c) % 256 for c in text[:max_len]]
  if len(tokens) < max_len:
    tokens = tokens + [0] * (max_len - len(tokens))
  return torch.tensor(tokens, dtype=torch.long)


@register_dataset("multimodal/shapes")
@register_dataset("multimodal/clip_shapes")
class MultimodalShapesDataset(SyntheticDataset):
  """Procedurally generates paired (RGB Image, Natural Text Caption) samples.

  Designed for evaluating CLIP-like dual encoders, cross-modal attention,
  and compositional attribute-binding (color, shape, spatial position).

  Each sample contains:
    - `image`: 3-channel RGB image with multiple colored shapes in distinct quadrants.
    - `input_ids`: 1D long tensor token representation of the descriptive caption.
    - `caption`: Ground-truth descriptive string (e.g. 'a red circle in the top-left').
    - `negative_captions`: Dict containing:
      * 'attr_negative': Attribute-swapped negative caption ('green circle').
      * 'pos_negative': Spatial-swapped negative caption ('red circle in the bottom-right').
  """

  def __init__(
    self,
    num_samples: int = 1000,
    img_size: int = 32,
    num_objects: int = 2,
    max_text_len: int = 64,
    seed: int = 42,
  ):
    """Initializes multimodal shapes dataset.

    Args:
      num_samples: Total number of paired samples (default: 1000).
      img_size: Height and width of the output image canvas (default: 32).
      num_objects: Number of distinct colored shapes per canvas (default: 2, max 4).
      max_text_len: Maximum token sequence length for text representation (default: 64).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._img_size = img_size
    self._num_objects = min(max(1, num_objects), 4)
    self._max_text_len = max_text_len

    generator = torch.Generator().manual_seed(seed)
    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    self._images = []
    self._tokens = []
    self._captions = []
    self._attr_negatives = []
    self._pos_negatives = []

    for _ in range(num_samples):
      canvas = torch.zeros((3, img_size, img_size), dtype=torch.float32)
      quad_indices = torch.randperm(4, generator=generator)[:self._num_objects].tolist()

      obj_descriptions = []
      alt_color_descriptions = []
      alt_pos_descriptions = []

      for q_idx in quad_indices:
        shape_idx = torch.randint(0, len(SHAPE_NAMES_MM), (1,), generator=generator).item()
        color_idx = torch.randint(0, len(COLOR_NAMES), (1,), generator=generator).item()
        alt_color_idx = (color_idx + 1) % len(COLOR_NAMES)

        shape_name = SHAPE_NAMES_MM[shape_idx]
        color_name = COLOR_NAMES[color_idx]
        alt_color_name = COLOR_NAMES[alt_color_idx]
        quad_name = QUADRANT_NAMES[q_idx]
        alt_quad_name = QUADRANT_NAMES[(q_idx + 2) % 4]

        cx, cy = QUADRANT_COORDS[q_idx]
        mask, _ = rasterize_shape(shape_idx, x_coords, y_coords, cx, cy, radius=0.2)

        r, g, b = COLOR_RGB[color_idx]
        color_tensor = torch.tensor([r, g, b], dtype=torch.float32).view(3, 1, 1)
        canvas = torch.where(mask.unsqueeze(0) > 0.5, color_tensor, canvas)

        obj_descriptions.append(f"{color_name} {shape_name} in the {quad_name}")
        alt_color_descriptions.append(f"{alt_color_name} {shape_name} in the {quad_name}")
        alt_pos_descriptions.append(f"{color_name} {shape_name} in the {alt_quad_name}")

      caption = "a " + " and a ".join(obj_descriptions)
      attr_neg = "a " + " and a ".join(alt_color_descriptions)
      pos_neg = "a " + " and a ".join(alt_pos_descriptions)

      token_ids = _tokenize_text(caption, max_len=max_text_len)

      self._images.append(canvas)
      self._tokens.append(token_ids)
      self._captions.append(caption)
      self._attr_negatives.append(attr_neg)
      self._pos_negatives.append(pos_neg)

    self._images_tensor = torch.stack(self._images, dim=0)
    self._tokens_tensor = torch.stack(self._tokens, dim=0)

  @property
  def images(self) -> torch.Tensor:
    """Returns stacked tensor of RGB images."""
    return self._images_tensor

  @property
  def tokens(self) -> torch.Tensor:
    """Returns stacked tensor of caption token sequences."""
    return self._tokens_tensor

  @property
  def captions(self) -> list[str]:
    """Returns list of ground-truth caption strings."""
    return self._captions

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing image, text token IDs, and string captions."""
    img = self._images_tensor[idx]
    tok = self._tokens_tensor[idx]
    cap = self._captions[idx]
    neg_attr = self._attr_negatives[idx]
    neg_pos = self._pos_negatives[idx]
    return DataSample(
      image=img,
      input_ids=tok,
      caption=cap,
      attr_negative=neg_attr,
      pos_negative=neg_pos,
    )

  def description(self) -> str:
    """Returns a short description of the dataset."""
    return f"Multimodal Shapes ({self._num_objects} objects, 3x{self._img_size}x{self._img_size})"


@register_dataset("multimodal/vqa")
class VisualQuestionAnsweringDataset(SyntheticDataset):
  """Procedurally generates visual question answering (VQA) triples (Image, Question, Answer).

  Tests multimodal reasoning: color identification, existence checking,
  and spatial relations between visual objects.
  """

  def __init__(
    self,
    num_samples: int = 1000,
    img_size: int = 32,
    max_text_len: int = 48,
    seed: int = 42,
  ):
    """Initializes synthetic VQA dataset.

    Args:
      num_samples: Total number of VQA items (default: 1000).
      img_size: Resolution of the image canvas (default: 32).
      max_text_len: Maximum token sequence length for question (default: 48).
      seed: Random seed for reproducible generation (default: 42).
    """
    super().__init__(num_samples=num_samples, seed=seed)
    self._img_size = img_size
    self._max_text_len = max_text_len

    generator = torch.Generator().manual_seed(seed)
    y_coords, x_coords = torch.meshgrid(
      torch.linspace(-1.0, 1.0, img_size),
      torch.linspace(-1.0, 1.0, img_size),
      indexing="ij",
    )

    self._images = []
    self._question_tokens = []
    self._questions = []
    self._answers = []
    self._answer_ids = []

    # Vocabulary for answers: 0..5 colors, 6: yes, 7: no
    answer_vocab = {c: i for i, c in enumerate(COLOR_NAMES)}
    answer_vocab["yes"] = 6
    answer_vocab["no"] = 7

    for _ in range(num_samples):
      canvas = torch.zeros((3, img_size, img_size), dtype=torch.float32)
      shape_a = torch.randint(0, len(SHAPE_NAMES_MM), (1,), generator=generator).item()
      color_a = torch.randint(0, len(COLOR_NAMES), (1,), generator=generator).item()
      cx_a, cy_a = -0.35, 0.0
      mask_a, _ = rasterize_shape(shape_a, x_coords, y_coords, cx_a, cy_a, radius=0.25)
      r_a, g_a, b_a = COLOR_RGB[color_a]
      canvas = torch.where(
        mask_a.unsqueeze(0) > 0.5,
        torch.tensor([r_a, g_a, b_a]).view(3, 1, 1),
        canvas,
      )

      # Shape B on the right
      shape_b = (shape_a + 1) % len(SHAPE_NAMES_MM)
      color_b = (color_a + 2) % len(COLOR_NAMES)
      cx_b, cy_b = 0.35, 0.0
      mask_b, _ = rasterize_shape(shape_b, x_coords, y_coords, cx_b, cy_b, radius=0.25)
      r_b, g_b, b_b = COLOR_RGB[color_b]
      canvas = torch.where(
        mask_b.unsqueeze(0) > 0.5,
        torch.tensor([r_b, g_b, b_b]).view(3, 1, 1),
        canvas,
      )

      # Pick question type: 0: What color is shape A? 1: Is there a circle?
      q_type = torch.randint(0, 2, (1,), generator=generator).item()
      if q_type == 0:
        question = f"what color is the {SHAPE_NAMES_MM[shape_a]}?"
        answer = COLOR_NAMES[color_a]
      else:
        question = f"is there a {SHAPE_NAMES_MM[shape_a]} on the canvas?"
        answer = "yes"

      ans_id = answer_vocab[answer]
      q_tokens = _tokenize_text(question, max_len=max_text_len)

      self._images.append(canvas)
      self._question_tokens.append(q_tokens)
      self._questions.append(question)
      self._answers.append(answer)
      self._answer_ids.append(ans_id)

    self._images_tensor = torch.stack(self._images, dim=0)
    self._question_tokens_tensor = torch.stack(self._question_tokens, dim=0)
    self._answer_ids_tensor = torch.tensor(self._answer_ids, dtype=torch.long)

  def __getitem__(self, idx: int) -> DataSample:
    """Returns a DataSample containing image, question, and ground-truth answer."""
    return DataSample(
      image=self._images_tensor[idx],
      question_ids=self._question_tokens_tensor[idx],
      answer_id=self._answer_ids_tensor[idx],
      question=self._questions[idx],
      answer=self._answers[idx],
    )

  def description(self) -> str:
    """Returns a description of the VQA task."""
    return f"Synthetic VQA (3x{self._img_size}x{self._img_size})"
