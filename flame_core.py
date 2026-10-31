"""Chapter 3: equations (1)-(7), LoQDC, and layer-ranking pruning."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")


@dataclass
class Config:
    bits: int = 4
    rank: int = 8
    lora_alpha: float = 32.0
    dropout: float = 0.05
    qa_alpha: float = 0.01
    loss_mode: str = "residual"
    beta1: float = 0.85
    beta2: float = 0.85
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 3
    init_fraction: float = 0.1
    init_epochs: int = 1
    lambda_update: str = "epoch"
    max_length: int = 512
    batch_size: int = 1
    seed: int = 42
    norm: str = "fro"
    cache_weights: bool = True

    def validate(self):
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4 or 8")
        if self.rank < 1 or self.epochs < 1 or self.init_epochs < 1:
            raise ValueError("rank and epochs must be positive")
        if not 0 < self.init_fraction <= 1:
            raise ValueError("init_fraction must be in (0,1]")
        if not 0 <= self.dropout < 1 or not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("invalid dropout or EMA coefficients")
        if self.qa_alpha < 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimization parameters")
        if self.loss_mode not in ("literal", "residual") or self.lambda_update not in ("epoch", "step"):
            raise ValueError("invalid loss mode or update frequency")
        if self.norm not in ("fro", "spectral") or self.batch_size < 1 or self.max_length < 1:
            raise ValueError("invalid norm, batch size or sequence length")

    def to_dict(self):
        return asdict(self)


def matrix_norm(x: Tensor, mode: str) -> Tensor:
    return torch.linalg.matrix_norm(x.float(), ord="fro" if mode == "fro" else 2)


def quantize_absmax(weight: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    """Equations (1)-(2); one scale per projection matrix, ties-to-even rounding."""
    if bits not in (2, 4, 8):
        raise ValueError("only 2/4/8-bit signed uniform quantization is supported")
    if not torch.isfinite(weight).all():
        raise ValueError("non-finite weight")
    w = weight.float()
    max_abs = w.abs().max()
    scale = torch.where(max_abs > 0, (2 ** (bits - 1) - 1) / max_abs, torch.ones_like(max_abs))
    q = torch.round(scale * w).clamp(-2 ** (bits - 1), 2 ** (bits - 1) - 1).to(torch.int8)
    return q, scale


def pack_signed(q: Tensor, bits: int) -> Tensor:
    """Pack signed 2/4/8-bit values, without changing the quantization equation."""
    per_byte = 8 // bits
    unsigned = (q.reshape(-1).to(torch.int16) + 2 ** (bits - 1)).to(torch.uint8)
    pad = (-unsigned.numel()) % per_byte
    if pad:
        unsigned = F.pad(unsigned, (0, pad))
    groups = unsigned.reshape(-1, per_byte)
    packed = torch.zeros(groups.shape[0], dtype=torch.uint8, device=q.device)
    for i in range(per_byte):
        packed |= groups[:, i] << (i * bits)
    return packed


def unpack_signed(packed: Tensor, bits: int, shape: tuple[int, ...]) -> Tensor:
    per_byte = 8 // bits
    groups = [(packed >> (i * bits)) & (2 ** bits - 1) for i in range(per_byte)]
    unsigned = torch.stack(groups, dim=1).reshape(-1)[:math.prod(shape)]
    return (unsigned.to(torch.int16) - 2 ** (bits - 1)).to(torch.int8).reshape(shape)


class QuantizedLoRALinear(nn.Module):
    """Frozen integer base plus scaled A@B; matrix orientation follows equation (6)."""
    def __init__(self, linear: nn.Linear, cfg: Config):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bits, self.norm_mode = cfg.bits, cfg.norm
        self.scaling = cfg.lora_alpha / cfg.rank
        self.cache_weights = cfg.cache_weights
        if linear.weight.is_meta:
            count = self.in_features * self.out_features
            packed = torch.empty(math.ceil(count / (8 // cfg.bits)), dtype=torch.uint8, device="meta")
            scale = delta = torch.empty((), device="meta")
            dequant = reference = None
        else:
            q, scale = quantize_absmax(linear.weight.detach(), cfg.bits)
            packed = pack_signed(q, cfg.bits)
            dequant_float = q.float() / scale
            delta = matrix_norm(linear.weight.detach().float() - dequant_float, cfg.norm)
            dequant = dequant_float.to(linear.weight.dtype)
            reference = linear.weight.detach().float().clone() if cfg.loss_mode == "residual" else None
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.register_buffer("delta", delta)
        self.register_buffer("cached_weight", dequant if cfg.cache_weights else None, persistent=False)
        self.register_buffer("reference", reference, persistent=False)
        self.register_buffer("bias", linear.bias.detach().clone() if linear.bias is not None else None)
        self.A = nn.Parameter(torch.empty(self.in_features, cfg.rank, device=linear.weight.device,
                                         dtype=linear.weight.dtype))
        self.B = nn.Parameter(torch.zeros(cfg.rank, self.out_features, device=linear.weight.device,
                                         dtype=linear.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.dropout = nn.Dropout(cfg.dropout)

    def dequantized_weight(self) -> Tensor:
        if self.cached_weight is not None:
            return self.cached_weight
        return unpack_signed(self.packed, self.bits, (self.out_features, self.in_features)).float() / self.scale

    def forward(self, x: Tensor) -> Tensor:
        w = self.dequantized_weight().to(x.dtype)
        return F.linear(x, w, self.bias) + (self.dropout(x) @ self.A @ self.B) * self.scaling

    def discrepancy(self, mode: str) -> Tensor:
        if mode == "literal":
            return self.delta
        if self.reference is None:
            raise RuntimeError("residual mode requires reference weights retained at installation")
        correction = (self.A.float() @ self.B.float()).T * self.scaling
        return matrix_norm(self.reference - self.dequantized_weight().float() - correction, self.norm_mode)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        w = self.dequantized_weight() + (self.A @ self.B).T * self.scaling
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None,
                           device=w.device, dtype=self.A.dtype)
        linear.weight.copy_(w)
        if self.bias is not None:
            linear.bias.copy_(self.bias)
        return linear


class ByteTokenizer:
    """Offline smoke-test tokenizer; not a substitute for CodeLlama's tokenizer."""
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, code: str, max_length: int) -> list[int]:
        return [b + 2 for b in code.encode("utf-8")[:max_length - 1]] + [1]

    def save_pretrained(self, path):
        import json
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "byte_tokenizer.json").write_text(json.dumps({"type": "byte", "vocab_size": 258}),
                                                     encoding="utf-8")


class TinyLayer(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, 2 * hidden, bias=False)
        self.gate_proj = nn.Linear(hidden, 2 * hidden, bias=False)
        self.down_proj = nn.Linear(2 * hidden, hidden, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor):
        b, t, h = x.shape
        v = self.attn_norm(x)
        q, k, val = [proj(v).reshape(b, t, self.heads, h // self.heads).transpose(1, 2)
                     for proj in (self.q_proj, self.k_proj, self.v_proj)]
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        mask = causal[None, None] & attention_mask[:, None, None, :].bool()
        attn = F.scaled_dot_product_attention(q, k, val, attn_mask=mask)
        x = x + self.o_proj(attn.transpose(1, 2).reshape(b, t, h))
        v = self.ffn_norm(x)
        return x + self.down_proj(F.silu(self.gate_proj(v)) * self.up_proj(v))


class TinyClassifier(nn.Module):
    def __init__(self, layers=4, hidden=32, heads=4, max_length=512):
        super().__init__()
        self.settings = dict(layers=layers, hidden=hidden, heads=heads, max_length=max_length)
        self.embedding = nn.Embedding(258, hidden, padding_idx=0)
        self.position = nn.Embedding(max_length, hidden)
        self.layers = nn.ModuleList([TinyLayer(hidden, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden)
        self.score = nn.Linear(hidden, 2, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embedding(input_ids) + self.position(positions)
        for layer in self.layers:
            x = layer(x, attention_mask)
        last = attention_mask.sum(dim=1) - 1
        return self.score(self.norm(x)[torch.arange(x.shape[0], device=x.device), last])


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    if isinstance(model, TinyClassifier):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError("expected a TinyClassifier or Llama/CodeLlama sequence classifier")


def set_decoder_layers(model: nn.Module, layers: nn.ModuleList):
    if isinstance(model, TinyClassifier):
        model.layers = layers
        model.settings["layers"] = len(layers)
    else:
        model.model.layers = layers
        model.config.num_hidden_layers = len(layers)
    for i, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = i


def quant_modules(layer: nn.Module) -> list[QuantizedLoRALinear]:
    return [m for m in layer.modules() if isinstance(m, QuantizedLoRALinear)]


def install_loqdc(model: nn.Module, cfg: Config):
    cfg.validate()
    for param in model.parameters():
        param.requires_grad_(False)
    for index, layer in enumerate(decoder_layers(model)):
        found = []
        for name, module in list(layer.named_modules()):
            if name.split(".")[-1] in TARGETS and isinstance(module, nn.Linear):
                parent_name, _, attr = name.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(parent, attr, QuantizedLoRALinear(module, cfg))
                found.append(attr)
        if sorted(found) != sorted(TARGETS):
            raise ValueError(f"layer {index}: expected seven projection matrices, found {found}")
    for param in model.score.parameters():
        param.requires_grad_(True)


class ImportanceTracker:
    """AdaLoRA-style EMAs: I=|parameter*gradient|, score=I_bar*U_bar."""
    def __init__(self, model: nn.Module, cfg: Config):
        self.model, self.beta1, self.beta2 = model, cfg.beta1, cfg.beta2
        self.state: dict[int, tuple[Tensor, Tensor]] = {}

    @torch.no_grad()
    def update(self):
        for layer in decoder_layers(self.model):
            for module in quant_modules(layer):
                for param in (module.A, module.B):
                    if param.grad is None:
                        continue
                    instant = (param.float() * param.grad.float()).abs()
                    smooth, uncertainty = self.state.get(id(param), (torch.zeros_like(instant),
                                                                    torch.zeros_like(instant)))
                    smooth = self.beta1 * smooth + (1 - self.beta1) * instant
                    uncertainty = self.beta2 * uncertainty + (1 - self.beta2) * (instant - smooth).abs()
                    self.state[id(param)] = smooth, uncertainty

    @torch.no_grad()
    def scores(self) -> Tensor:
        rows = []
        for layer in decoder_layers(self.model):
            score = torch.zeros((), device=next(self.model.parameters()).device)
            for module in quant_modules(layer):
                for param in (module.A, module.B):
                    if id(param) in self.state:
                        smooth, uncertainty = self.state[id(param)]
                        score = score + (smooth * uncertainty).mean()
            rows.append(score)
        return torch.stack(rows)


def layer_discrepancies(model: nn.Module, mode: str) -> Tensor:
    return torch.stack([torch.stack([m.discrepancy(mode) for m in quant_modules(layer)]).sum()
                        for layer in decoder_layers(model)])


def qa_loss(ce: Tensor, model: nn.Module, lambdas: Tensor, cfg: Config) -> tuple[Tensor, Tensor]:
    penalty = cfg.qa_alpha * torch.sum(lambdas.detach() * layer_discrepancies(model, cfg.loss_mode))
    return ce + penalty, penalty


def logits_of(model: nn.Module, ids: Tensor, mask: Tensor) -> Tensor:
    output = model(input_ids=ids, attention_mask=mask)
    return output if isinstance(output, Tensor) else output.logits


def binary_metrics(labels: list[int], predictions: list[int]) -> dict:
    if not labels or len(labels) != len(predictions):
        raise ValueError("non-empty, aligned labels and predictions required")
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(n=len(labels), accuracy=(tp + tn) / len(labels), precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                tp=tp, tn=tn, fp=fp, fn=fn)


def select_layers(model: nn.Module, scores: list[float], evaluate: Callable[[], dict],
                  max_drop: float = 0.01, remove_layers: int | None = None) -> dict:
    """Top-k by score, but execution remains in original order; use validation only."""
    original = list(decoder_layers(model))
    if len(scores) != len(original) or not all(math.isfinite(s) for s in scores):
        raise ValueError("one finite importance score per layer required")
    if max_drop < 0 or remove_layers is not None and not 0 <= remove_layers < len(original):
        raise ValueError("invalid accuracy budget or number of removed layers")
    ranked = sorted(range(len(original)), key=lambda i: (-scores[i], i))
    baseline = evaluate()
    counts = [len(original) - remove_layers] if remove_layers is not None else list(range(1, len(original) + 1))
    trials, chosen = [], None
    try:
        for k in counts:
            kept = sorted(ranked[:k])
            set_decoder_layers(model, nn.ModuleList([original[i] for i in kept]))
            metrics = evaluate()
            accepted = baseline["accuracy"] - metrics["accuracy"] <= max_drop + 1e-12
            trial = dict(k=k, kept_original_indices=kept, metrics=metrics, accepted=accepted)
            trials.append(trial)
            if accepted:
                chosen = trial
                break
        if chosen is None:
            raise RuntimeError("requested top-k exceeds validation accuracy-loss budget; no model exported")
    except BaseException:
        set_decoder_layers(model, nn.ModuleList(original))
        raise
    return dict(baseline=baseline, max_drop_absolute=max_drop, selected=chosen, trials=trials)


@torch.no_grad()
def merge_adapters(model: nn.Module):
    for layer in decoder_layers(model):
        for name, module in list(layer.named_modules()):
            if isinstance(module, QuantizedLoRALinear):
                parent_name, _, attr = name.rpartition(".")
                parent = layer.get_submodule(parent_name) if parent_name else layer
                setattr(parent, attr, module.merged_linear())
