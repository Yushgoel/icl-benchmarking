import numpy as np
import torch


def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    try:
        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.item()
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} not serializable")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
