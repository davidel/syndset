"""Utilities for attaching and managing forward and backward hooks."""

import collections

import torch


class ForwardActivationHook:
  """Context manager to intercept and record layer forward activations.

  Useful for inspecting hidden representations, checking for dead channels,
  and calculating effective rank across layers.
  """

  def __init__(self, model, target_types=None):
    """Initializes the forward activation recorder.

    Args:
      model: The torch.nn.Module to monitor.
      target_types: An optional tuple or list of module classes to monitor.
        If None, records all leaf submodules that have parameters.
    """
    self.model = model
    self.target_types = target_types
    self.activations = collections.OrderedDict()
    self._handles = []

  def __enter__(self):
    """Registers forward hooks on targeted submodules."""
    self.activations.clear()
    for name, module in self.model.named_modules():
      if module == self.model:
        continue
      if self._should_hook(module):
        handle = module.register_forward_hook(self._make_hook(name))
        self._handles.append(handle)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Removes all registered hooks to avoid memory leaks."""
    self.remove()

  def _should_hook(self, module):
    """Determines whether a submodule should be hooked.

    Args:
      module: A candidate torch.nn.Module.

    Returns:
      True if the module matches the target criteria, False otherwise.
    """
    if self.target_types is not None:
      return isinstance(module, tuple(self.target_types))
    has_params = any(p.requires_grad for p in module.parameters(recurse=False))
    has_children = len(list(module.children())) > 0
    return has_params or not has_children

  def _make_hook(self, name):
    """Creates a hook closure bound to the layer name.

    Args:
      name: The string name of the layer.

    Returns:
      A hook function matching the PyTorch forward hook signature.
    """

    def hook(module, inputs, output):
      if isinstance(output, torch.Tensor):
        self.activations[name] = output.detach()
      elif isinstance(output, (tuple, list)):
        for idx, item in enumerate(output):
          if isinstance(item, torch.Tensor):
            self.activations[name + f"[{idx}]"] = item.detach()
            break

    return hook

  def remove(self):
    """Removes all active hook handles."""
    for handle in self._handles:
      handle.remove()
    self._handles.clear()


class BackwardGradientHook:
  """Context manager to intercept and record tensor gradients during backward pass.

  Attaches hooks to leaf parameters to observe gradient statistics directly
  at the parameter level as backpropagation executes.
  """

  def __init__(self, model):
    """Initializes the gradient recorder.

    Args:
      model: The torch.nn.Module whose parameters to track.
    """
    self.model = model
    self.gradients = collections.OrderedDict()
    self._handles = []

  def __enter__(self):
    """Registers tensor hooks on all trainable parameters."""
    self.gradients.clear()
    for name, param in self.model.named_parameters():
      if param.requires_grad:
        handle = param.register_hook(self._make_hook(name))
        self._handles.append(handle)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Removes all registered parameter hooks."""
    self.remove()

  def _make_hook(self, name):
    """Creates a tensor hook closure for a named parameter.

    Args:
      name: The parameter name.

    Returns:
      A hook function taking grad and recording it.
    """

    def hook(grad):
      self.gradients[name] = grad.detach()

    return hook

  def remove(self):
    """Removes all active hook handles."""
    for handle in self._handles:
      handle.remove()
    self._handles.clear()
