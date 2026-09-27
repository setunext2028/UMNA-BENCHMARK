"""Training implementations for UMNA interference probe.

Four methods compared:
- UMNA: Train wing transform + head through frozen Core (Core never updates)
- LoRA: Add low-rank adapters to Core weights, train only LoRA params
- Adapter: Add bottleneck adapters between Core layers, train only adapter params
- FullFT: Full fine-tuning of Core + wing parameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.model import FrozenCore, forward, backward, params_sha256, WIDTHS
from core.tasks import INPUT_DIM
from core.firewall import (
    _sigmoid,
    _init_wing_params,
    _init_head_params,
    train_wing_through_core,
    WingArtifact,
)


WING_WIDTH = 16
WING_STEPS = 600
WING_BATCH = 64
WING_LR = 0.05
WING_WEIGHT_DECAY = 1e-4
WING_SIGMOID_CLIP = 30.0

LORA_RANK = 4
ADAPTER_BOTTLENECK = 4


def _sigmoid_local(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -WING_SIGMOID_CLIP, WING_SIGMOID_CLIP)))


@dataclass(frozen=True)
class TrainedWing:
    """Trained wing parameters that can be used for evaluation."""
    transform: dict[str, np.ndarray]
    head: dict[str, np.ndarray]
    core_epoch: str
    core_sha256: str
    core_params: dict[str, np.ndarray] | None = None  # For FullFT: modified core params


def _predict_wing(
    wing_params: dict[str, np.ndarray],
    head_params: dict[str, np.ndarray],
    core: FrozenCore,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Forward pass through wing -> core -> head. Returns logits, latent, pre_activation, cache."""
    pre = x @ wing_params["w"] + wing_params["b"]
    features = np.maximum(pre, 0.0)
    latent, cache = core.forward(features)
    logits = latent @ head_params["w"] + head_params["b"]
    return logits, latent, pre, cache


def train_umna(
    core: FrozenCore,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 1,
    steps: int = WING_STEPS,
    batch_size: int = WING_BATCH,
    learning_rate: float = WING_LR,
    weight_decay: float = WING_WEIGHT_DECAY,
) -> TrainedWing:
    """UMNA method: train wing transform + head through frozen Core."""
    artifact = train_wing_through_core(
        core=core,
        name=name,
        x=x,
        y=y,
        seed=seed,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    return TrainedWing(
        transform=artifact.transform,
        head=artifact.head,
        core_epoch=artifact.core_epoch,
        core_sha256=artifact.core_sha256,
    )


def train_lora(
    core: FrozenCore,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 1,
    steps: int = WING_STEPS,
    batch_size: int = WING_BATCH,
    learning_rate: float = WING_LR,
    weight_decay: float = WING_WEIGHT_DECAY,
    rank: int = LORA_RANK,
) -> TrainedWing:
    """LoRA method: train low-rank adapters on Core weights + wing."""
    
    if core.epoch != "1.x":
        raise ValueError(f"core epoch {core.epoch!r} is not 1.x")
    
    rng = np.random.default_rng(seed)
    core_before = core.sha256
    
    wing = _init_wing_params(rng)
    head = _init_head_params(rng)
    
    # LoRA params: A @ B where A is (in_dim, rank), B is (rank, out_dim)
    lora_w0_A = rng.uniform(-1.0, 1.0, size=(WIDTHS[0], rank)).astype(np.float32) * 0.01
    lora_w0_B = np.zeros((rank, WIDTHS[1]), dtype=np.float32)
    lora_w1_A = rng.uniform(-1.0, 1.0, size=(WIDTHS[1], rank)).astype(np.float32) * 0.01
    lora_w1_B = np.zeros((rank, WIDTHS[2]), dtype=np.float32)
    
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32).reshape(-1, 1)
    
    for _ in range(steps):
        rows = rng.integers(0, len(x), size=batch_size)
        batch_x = x[rows]
        batch_y = y[rows]
        
        pre = batch_x @ wing["w"] + wing["b"]
        features = np.maximum(pre, 0.0)
        
        # Core forward with LoRA
        pre_w0 = features @ core.params["w0"] + core.params["b0"]
        lora_w0_delta = features @ (lora_w0_A @ lora_w0_B)
        hidden_w0 = np.maximum(pre_w0 + lora_w0_delta, 0.0)
        
        pre_w1 = hidden_w0 @ core.params["w1"] + core.params["b1"]
        lora_w1_delta = hidden_w0 @ (lora_w1_A @ lora_w1_B)
        latent = pre_w1 + lora_w1_delta
        
        logits = latent @ head["w"] + head["b"]
        grad_logits = (_sigmoid_local(logits) - batch_y) / batch_size
        
        # Head gradients
        grad_head_w = latent.T @ grad_logits + weight_decay * head["w"]
        grad_head_b = grad_logits.sum(axis=0)
        grad_latent = grad_logits @ head["w"].T  # (batch, 8)
        
        # LoRA w1 gradients: w1_eff = w1 + A1 @ B1
        # dL/dW1_eff = hidden_w0.T @ grad_latent  -> (12, 8)
        grad_W1_eff = hidden_w0.T @ grad_latent
        
        # dL/dB1 = A1.T @ dL/dW1_eff  -> (rank, 8)
        grad_lora_w1_B = lora_w1_A.T @ grad_W1_eff
        # dL/dA1 = dL/dW1_eff @ B1.T  -> (12, rank)
        grad_lora_w1_A = grad_W1_eff @ lora_w1_B.T
        
        # Backprop through w1 and LoRA w1
        # grad_pre_w1 = grad_latent (since pre_w1 = latent for linear layer)
        # grad_hidden_w0 = grad_pre_w1 @ w1.T + grad_pre_w1 @ B1.T @ A1.T
        grad_hidden_w0 = grad_latent @ core.params["w1"].T + grad_latent @ lora_w1_B.T @ lora_w1_A.T
        grad_hidden_w0 = grad_hidden_w0 * (pre_w0 > 0.0)
        
        # LoRA w0 gradients: w0_eff = w0 + A0 @ B0
        # dL/dpre_w0 = grad_hidden_w0 * (pre_w0 > 0)  -> (batch, 12)
        grad_pre_w0 = grad_hidden_w0 * (pre_w0 > 0.0)
        
        # dL/dW0_eff = features.T @ grad_pre_w0  -> (16, 12)
        grad_W0_eff = features.T @ grad_pre_w0
        
        # dL/dB0 = A0.T @ dL/dW0_eff  -> (rank, 12)
        grad_lora_w0_B = lora_w0_A.T @ grad_W0_eff
        # dL/dA0 = dL/dW0_eff @ B0.T  -> (16, rank)
        grad_lora_w0_A = grad_W0_eff @ lora_w0_B.T
        
        # Backprop through w0 and LoRA w0 to features
        # dL/dfeatures = dL/dpre_w0 @ w0.T + dL/dpre_w0 @ B0.T @ A0.T
        grad_features = grad_pre_w0 @ core.params["w0"].T + grad_pre_w0 @ lora_w0_B.T @ lora_w0_A.T
        
        # Wing gradients
        grad_pre = grad_features
        grad_pre = grad_pre * (pre > 0.0)
        grad_w = batch_x.T @ grad_pre + weight_decay * wing["w"]
        grad_b = grad_pre.sum(axis=0)
        
        # Update wing + head + LoRA (NOT Core params)
        wing["w"] -= learning_rate * grad_w
        wing["b"] -= learning_rate * grad_b
        head["w"] -= learning_rate * grad_head_w
        head["b"] -= learning_rate * grad_head_b
        
        lora_w0_A -= learning_rate * grad_lora_w0_A
        lora_w0_B -= learning_rate * grad_lora_w0_B
        lora_w1_A -= learning_rate * grad_lora_w1_A
        lora_w1_B -= learning_rate * grad_lora_w1_B
    
    if core.sha256 != core_before:
        raise RuntimeError("core hash changed during LoRA training")
    
    return TrainedWing(
        transform=wing,
        head=head,
        core_epoch=core.epoch,
        core_sha256=core.sha256,
    )


def train_adapter(
    core: FrozenCore,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 1,
    steps: int = WING_STEPS,
    batch_size: int = WING_BATCH,
    learning_rate: float = WING_LR,
    weight_decay: float = WING_WEIGHT_DECAY,
    bottleneck: int = ADAPTER_BOTTLENECK,
) -> TrainedWing:
    """Adapter method: train bottleneck adapters between Core layers + wing."""
    
    if core.epoch != "1.x":
        raise ValueError(f"core epoch {core.epoch!r} is not 1.x")
    
    rng = np.random.default_rng(seed)
    core_before = core.sha256
    
    wing = _init_wing_params(rng)
    head = _init_head_params(rng)
    
    # Adapter between w0 and w1: 12 -> bottleneck -> 12
    adapter_down = rng.uniform(-1.0, 1.0, size=(WIDTHS[1], bottleneck)).astype(np.float32) * 0.01
    adapter_up = np.zeros((bottleneck, WIDTHS[1]), dtype=np.float32)
    adapter_bias = np.zeros(bottleneck, dtype=np.float32)
    
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32).reshape(-1, 1)
    
    for _ in range(steps):
        rows = rng.integers(0, len(x), size=batch_size)
        batch_x = x[rows]
        batch_y = y[rows]
        
        pre = batch_x @ wing["w"] + wing["b"]
        features = np.maximum(pre, 0.0)
        
        # Core layer 0
        pre_w0 = features @ core.params["w0"] + core.params["b0"]
        hidden_w0 = np.maximum(pre_w0, 0.0)
        
        # Adapter
        adapter_hidden = hidden_w0 @ adapter_down + adapter_bias
        adapter_hidden = np.maximum(adapter_hidden, 0.0)
        adapter_out = adapter_hidden @ adapter_up
        hidden_w0_adapted = hidden_w0 + adapter_out
        
        # Core layer 1
        pre_w1 = hidden_w0_adapted @ core.params["w1"] + core.params["b1"]
        latent = pre_w1
        
        logits = latent @ head["w"] + head["b"]
        grad_logits = (_sigmoid_local(logits) - batch_y) / batch_size
        
        # Head gradients
        grad_head_w = latent.T @ grad_logits + weight_decay * head["w"]
        grad_head_b = grad_logits.sum(axis=0)
        grad_latent = grad_logits @ head["w"].T
        
        # Backprop through w1
        grad_hidden_w0_adapted = grad_latent @ core.params["w1"].T
        
        # Adapter gradients
        grad_adapter_out = grad_hidden_w0_adapted
        grad_adapter_up = adapter_hidden.T @ grad_adapter_out
        grad_adapter_hidden = grad_adapter_out @ adapter_up.T
        grad_adapter_hidden = grad_adapter_hidden * (adapter_hidden > 0.0)
        grad_adapter_down = hidden_w0.T @ grad_adapter_hidden
        grad_adapter_bias = grad_adapter_hidden.sum(axis=0)
        
        grad_hidden_w0 = grad_hidden_w0_adapted + grad_adapter_hidden @ adapter_down.T
        grad_hidden_w0 = grad_hidden_w0 * (pre_w0 > 0.0)
        
        # Wing gradients
        grad_pre = grad_hidden_w0 @ core.params["w0"].T
        grad_pre = grad_pre * (pre > 0.0)
        grad_w = batch_x.T @ grad_pre + weight_decay * wing["w"]
        grad_b = grad_pre.sum(axis=0)
        
        # Update wing + head + adapter (NOT Core params)
        wing["w"] -= learning_rate * grad_w
        wing["b"] -= learning_rate * grad_b
        head["w"] -= learning_rate * grad_head_w
        head["b"] -= learning_rate * grad_head_b
        
        adapter_down -= learning_rate * grad_adapter_down
        adapter_up -= learning_rate * grad_adapter_up
        adapter_bias -= learning_rate * grad_adapter_bias
    
    if core.sha256 != core_before:
        raise RuntimeError("core hash changed during Adapter training")
    
    return TrainedWing(
        transform=wing,
        head=head,
        core_epoch=core.epoch,
        core_sha256=core.sha256,
    )


def train_fullft(
    core: FrozenCore,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 1,
    steps: int = WING_STEPS,
    batch_size: int = WING_BATCH,
    learning_rate: float = WING_LR,
    weight_decay: float = WING_WEIGHT_DECAY,
) -> TrainedWing:
    """Full fine-tuning: train Core + wing + head (Core becomes writable copy)."""
    
    rng = np.random.default_rng(seed)
    
    # Make a writable copy of core params for full fine-tuning
    core_params = {k: v.copy() for k, v in core.params.items()}
    for arr in core_params.values():
        arr.setflags(write=True)
    
    wing = _init_wing_params(rng)
    head = _init_head_params(rng)
    
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32).reshape(-1, 1)
    
    for _ in range(steps):
        rows = rng.integers(0, len(x), size=batch_size)
        batch_x = x[rows]
        batch_y = y[rows]
        
        pre = batch_x @ wing["w"] + wing["b"]
        features = np.maximum(pre, 0.0)
        
        pre_w0 = features @ core_params["w0"] + core_params["b0"]
        hidden_w0 = np.maximum(pre_w0, 0.0)
        
        pre_w1 = hidden_w0 @ core_params["w1"] + core_params["b1"]
        latent = pre_w1
        
        logits = latent @ head["w"] + head["b"]
        grad_logits = (_sigmoid_local(logits) - batch_y) / batch_size
        
        # Head gradients
        grad_head_w = latent.T @ grad_logits + weight_decay * head["w"]
        grad_head_b = grad_logits.sum(axis=0)
        grad_latent = grad_logits @ head["w"].T
        
        # Core w1 gradients
        grad_w1 = hidden_w0.T @ grad_latent + weight_decay * core_params["w1"]
        grad_b1 = grad_latent.sum(axis=0)
        grad_hidden_w0 = grad_latent @ core_params["w1"].T
        grad_hidden_w0 = grad_hidden_w0 * (pre_w0 > 0.0)
        
        # Core w0 gradients
        grad_w0 = features.T @ grad_hidden_w0 + weight_decay * core_params["w0"]
        grad_b0 = grad_hidden_w0.sum(axis=0)
        grad_pre_w0 = grad_hidden_w0 * (pre_w0 > 0.0)
        grad_features = grad_pre_w0 @ core_params["w0"].T
        
        # Wing gradients
        grad_pre = grad_features
        grad_pre = grad_pre * (pre > 0.0)
        grad_w = batch_x.T @ grad_pre + weight_decay * wing["w"]
        grad_b = grad_pre.sum(axis=0)
        
        # Update everything including Core
        wing["w"] -= learning_rate * grad_w
        wing["b"] -= learning_rate * grad_b
        head["w"] -= learning_rate * grad_head_w
        head["b"] -= learning_rate * grad_head_b
        
        core_params["w1"] -= learning_rate * grad_w1
        core_params["b1"] -= learning_rate * grad_b1
        core_params["w0"] -= learning_rate * grad_w0
        core_params["b0"] -= learning_rate * grad_b0
    
    # Verify core hash changed (it should for FullFT)
    new_sha = params_sha256(core_params)
    if new_sha == core.sha256:
        raise RuntimeError("FullFT should change core hash but didn't")
    
    return TrainedWing(
        transform=wing,
        head=head,
        core_epoch=core.epoch,
        core_sha256=new_sha,
        core_params=core_params,
    )


METHODS: dict[str, Callable] = {
    "UMNA": train_umna,
    "LoRA": train_lora,
    "Adapter": train_adapter,
    "FullFT": train_fullft,
}


def train_wing(
    method: str,
    core: FrozenCore,
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    **kwargs,
) -> TrainedWing:
    """Dispatch to the appropriate training method."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHODS.keys())}")
    return METHODS[method](core, name, x, y, **kwargs)


def evaluate_wing(
    trained: TrainedWing,
    core: FrozenCore,
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int | None = None,
) -> tuple[float, float]:
    """Evaluate a trained wing on data, return (accuracy, std_score)."""
    if n_samples is not None:
        x = x[:n_samples]
        y = y[:n_samples]
    correct = 0
    scores = []
    for i in range(len(x)):
        packet = np.array([x[i]], dtype=np.float32)
        logits, latent, pre, _ = _predict_wing(trained.transform, trained.head, core, packet)
        prob = float(_sigmoid_local(logits[0, 0]))
        pred = 1 if prob >= 0.5 else 0
        true_label = int(y[i])
        correct += (pred == true_label)
        scores.append(prob)
    accuracy = correct / len(x)
    return accuracy, float(np.std(scores))