"""General helpers shared by Carvana training, evaluation, and prediction."""

import importlib.util
import random
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    """Return the requested device, or automatically select CUDA when available."""
    if device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_parent_dir(path: Path | str) -> Path:
    """Create the parent directory for a file path and return the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_checkpoint(checkpoint_path: Path | str,
                    map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint saved by the project's training scripts."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # weights_only is supported by newer PyTorch versions. The fallback keeps
    # this helper compatible with older Kaggle/local PyTorch installations.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise TypeError("Expected the checkpoint to contain a dictionary")
    return checkpoint


def load_model_weights(model: torch.nn.Module,
                       checkpoint_path: Path | str,
                       device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load model weights and return checkpoint metadata.

    Supports both the project's checkpoint format (with model_state_dict)
    and a plain PyTorch state_dict.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise KeyError("Checkpoint does not contain a valid model_state_dict")
    model.load_state_dict(state_dict)
    return checkpoint


def import_model_class(model_file: Path | str, class_name: str) -> type:
    """Import a model class from a Python file path.

    This avoids hard-coding U-Net or LinkNet imports in shared evaluation and
    submission scripts. The selected model file should define the class at
    module scope and should not rely on relative imports from its own folder.
    """
    model_file = Path(model_file).resolve()
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    spec = importlib.util.spec_from_file_location("carvana_selected_model", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from: {model_file}")

    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, class_name, None)
    if model_class is None or not isinstance(model_class, type):
        raise AttributeError(
            f"Class {class_name!r} was not found in {model_file}. "
            "Check the class name in your model.py."
        )
    return model_class


def extract_logits(model_output: Any) -> torch.Tensor:
    """Normalize common segmentation-model outputs to a logits tensor."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, dict):
        if "out" in model_output and isinstance(model_output["out"], torch.Tensor):
            return model_output["out"]
        raise TypeError("Model output dictionary must contain a tensor under key 'out'")
    if isinstance(model_output, (tuple, list)) and model_output:
        if isinstance(model_output[0], torch.Tensor):
            return model_output[0]
    raise TypeError(
        "Model must return a logits tensor, a dict containing 'out', "
        "or a tuple/list whose first item is a tensor"
    )
