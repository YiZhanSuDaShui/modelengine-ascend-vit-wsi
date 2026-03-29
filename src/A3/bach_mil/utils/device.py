from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')

    # torch_npu registers `torch.npu` lazily on import.
    try:
        import torch_npu  # noqa: F401
    except Exception:
        torch_npu = None  # noqa: F841

    if hasattr(torch, 'npu') and torch.npu.is_available():
        dev = torch.device('npu:0')
        torch.npu.set_device(dev)
        return dev

    return torch.device('cpu')
