import timm
import torch.nn as nn
from typing import Dict, List
import numpy as np

# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_cfg: Dict):
    name = model_cfg["name"].lower()
    if name == "resnet20":
        model = timm.create_model("cifar_resnet20", num_classes=model_cfg.get("num_classes", 10), pretrained=False)
    else:
        model = timm.create_model(name, num_classes=model_cfg.get("num_classes", 10), pretrained=False)
    assert isinstance(model, nn.Module)
    return model

# ---------------------------------------------------------------------------
# Curve-compression helpers
# ---------------------------------------------------------------------------

def edcp_score(curve: List[float], gamma: float = 0.95) -> float:
    curve = np.asarray(curve, dtype=float)
    T = len(curve)
    weights = gamma ** (np.arange(T)[::-1])
    return float(np.dot(weights, curve))

def logistic_sigmoid_score(curve: List[float], m0: float, g0: float) -> float:
    curve = np.asarray(curve, dtype=float)
    t = np.arange(len(curve))
    weights = 1.0 / (1.0 + np.exp(-g0 * (t - m0)))
    return float(np.dot(weights, curve) / weights.sum())