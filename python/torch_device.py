"""
Selection of the torch device for training, UKF and NEMPC

Order: CUDA (NVIDIA GPU), MPS (Apple Metal GPU), CPU. The environment variable
ROD_PINN_DEVICE ("cuda", "mps" or "cpu") overrides the automatic choice.
"""

import os
import platform

import torch


def get_device(use_gpu=True, set_default=True, verbose=True):
    """Return the best available torch.device and optionally make it the default device."""
    requested = os.environ.get("ROD_PINN_DEVICE", "").lower()
    if requested:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ROD_PINN_DEVICE=cuda, but CUDA is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("ROD_PINN_DEVICE=mps, but MPS (Apple Metal) is not available")
        device = torch.device("cuda:0" if requested == "cuda" else requested)
    elif use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif use_gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
    if set_default:
        torch.set_default_device(device)
    if verbose:
        print(f"Using {device_description(device)}")
    return device


def device_description(device):
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return f"MPS (Apple Metal, {platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"


def synchronize(device):
    """Wait for all kernels on the device to finish, e.g. before reading a timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def to_device(tensor, device):
    """Move a tensor to the device; MPS has no float64, so float64 becomes float32 there."""
    if device.type == "mps" and tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
    return tensor.to(device=device)
