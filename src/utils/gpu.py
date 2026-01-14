"""GPU and model loading utilities."""

import torch
from typing import Optional


def get_device(gpu_id: int = 0) -> str:
    """
    Get the appropriate device string for PyTorch.

    Args:
        gpu_id: GPU index to use (default: 0)

    Returns:
        Device string (e.g., "cuda:0" or "cpu")
    """
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    return device


def get_torch_dtype(device: str) -> torch.dtype:
    """
    Get appropriate torch dtype based on device.

    Args:
        device: Device string

    Returns:
        torch.dtype (bfloat16 for CUDA, float32 for CPU)
    """
    if "cuda" in device:
        return torch.bfloat16
    return torch.float32


def load_sentence_transformer(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None
):
    """
    Load a SentenceTransformer model.

    Args:
        model_name: Model name or path
        device: Device to load model on

    Returns:
        SentenceTransformer model
    """
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = get_device()

    model = SentenceTransformer(model_name, device=device)
    return model


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
