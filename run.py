#!/usr/bin/env python3
"""Train/evaluate a chapter-3 FLAME reproduction; offline demo needs only PyTorch."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import time
import warnings

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import distributed_runtime as runtime
from data_io import (HFTokenizer, dump_json, dump_jsonl, load_records, loader,
                     split_records, synthetic_records)
from flame_core import (ByteTokenizer, Config, ImportanceTracker, QuantizedLoRALinear,
                        TinyClassifier, binary_metrics, decoder_layers, install_loqdc,
                        layer_discrepancies, logits_of, merge_adapters, qa_loss, select_layers)

FORMAT_VERSION = 1


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, records, tokenizer, cfg, device):
    was_training = model.training
    model.eval()
    labels, predictions, ce_sum = [], [], 0.0
    synchronize(device)
    start = time.perf_counter()
    try:
        for ids, mask, y in loader(records, tokenizer, cfg):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            logits = logits_of(model, ids, mask)
            ce_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
            labels.extend(y.tolist())
            predictions.extend(logits.argmax(dim=-1).tolist())
        synchronize(device)
    finally:
        model.train(was_training)
    metrics = binary_metrics(labels, predictions)
    metrics.update(cross_entropy=ce_sum / len(labels), elapsed_s=time.perf_counter() - start)
    by_language = {}
    for lang in sorted({row["language"] for row in records}):
        indices = [i for i, row in enumerate(records) if row["language"] == lang]
        by_language[lang] = binary_metrics([labels[i] for i in indices], [predictions[i] for i in indices])
    metrics["by_language"] = by_language
    return metrics


def fit(model, train, tokenizer, cfg, device, output):
    train_model = model
    if runtime.active():
        train_model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    tracker = ImportanceTracker(model, cfg)
    lambdas = torch.zeros(len(decoder_layers(model)), device=device)
    indices = random.Random(cfg.seed).sample(range(len(train)), max(1, round(len(train) * cfg.init_fraction)))
    initialization = [train[i] for i in indices]
    runtime.on_main(lambda: dump_json(output / "initialization_ids.json", [row["id"] for row in initialization]))
    history = []
    phases = [("initialization", initialization, cfg.init_epochs), ("compensation", train, cfg.epochs)]
    log_context = open(output / "training.jsonl", "w", encoding="utf-8") if runtime.rank() == 0 else nullcontext()
    with log_context as log:
        for phase, dataset, epochs in phases:
            for epoch in range(epochs):
                model.train()
                ce_total = loss_total = penalty_total = 0.0
                n = 0
                for ids, mask, labels in loader(dataset, tokenizer, cfg, shuffle=True,
                                                distributed=runtime.active(), epoch=epoch):
                    ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    ce = F.cross_entropy(logits_of(train_model, ids, mask).float(), labels)
                    if phase == "initialization":
                        loss, penalty = ce, ce.new_zeros(())
                    else:
                        loss, penalty = qa_loss(ce, model, lambdas, cfg)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite loss; no result exported")
                    loss.backward()
                    tracker.update()
                    optimizer.step()
                    if phase == "compensation" and cfg.lambda_update == "step":
                        lambdas = tracker.scores().sigmoid().detach()
                    size = labels.numel()
                    n += size
                    ce_total += ce.detach().item() * size
                    loss_total += loss.detach().item() * size
                    penalty_total += penalty.detach().item() * size
                scores = tracker.scores()
                if phase == "initialization" or cfg.lambda_update == "epoch":
                    lambdas = scores.sigmoid().detach()
                ce_total, loss_total, penalty_total, n = runtime.reduce_totals(
                    [ce_total, loss_total, penalty_total, n], device)
                record = dict(phase=phase, epoch=epoch + 1, samples=int(n), unique_samples=len(dataset),
                              world_size=runtime.world_size(), loss_mode=cfg.loss_mode, ce=ce_total / n,
                              qa_loss=loss_total / n, penalty=penalty_total / n,
                              layer_scores=scores.tolist(), lambdas=lambdas.tolist())
                history.append(record)
                if log is not None:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(f"{phase} epoch={epoch + 1} CE={record['ce']:.6f} "
                          f"penalty={record['penalty']:.6f} QALoss={record['qa_loss']:.6f}", flush=True)
    return tracker.scores().tolist(), history


def load_backbone(args, cfg, device):
    if args.tiny:
        return TinyClassifier(layers=args.tiny_layers, hidden=32, max_length=cfg.max_length), ByteTokenizer()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not args.model:
        raise ValueError("--model must be a local CodeLlama path or a Hugging Face checkpoint ID")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device.type == "cpu" else torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, num_labels=2, torch_dtype=dtype,
        attn_implementation="eager", trust_remote_code=False)
    if model.config.model_type != "llama":
        raise ValueError("paper backbone is CodeLlama; only llama model_type supported")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    return model, HFTokenizer(tokenizer)


def save_artifact(model, tokenizer, cfg, path, metadata):
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(path / "tokenizer")
    shape = model.settings if isinstance(model, TinyClassifier) else model.config.to_dict()
    payload = dict(format_version=FORMAT_VERSION, model_type="tiny" if isinstance(model, TinyClassifier) else "llama",
                   model_settings=shape, config=cfg.to_dict(), metadata=metadata,
                   state_dict={name: value.detach().cpu() for name, value in model.state_dict().items()})
    torch.save(payload, path / "model.pt")
    dump_json(path / "metadata.json", {k: v for k, v in payload.items() if k != "state_dict"})


def load_artifact(path, device):
    path = Path(path)
    artifact = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    if artifact["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported artifact format")
    cfg = Config(**artifact["config"])
    cfg.validate()
    with torch.device("meta"):
        if artifact["model_type"] == "tiny":
            model = TinyClassifier(**artifact["model_settings"])
        else:
            from transformers import AutoModelForSequenceClassification, LlamaConfig
            hf_cfg = LlamaConfig.from_dict(artifact["model_settings"])
            hf_cfg._attn_implementation = "eager"
            dtype = next(v.dtype for v in artifact["state_dict"].values() if v.is_floating_point())
            model = AutoModelForSequenceClassification.from_config(hf_cfg, torch_dtype=dtype)
        install_loqdc(model, replace(cfg, cache_weights=False))
    model.load_state_dict(artifact["state_dict"], strict=True, assign=True)
    if artifact["model_type"] == "llama":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        for name, module in list(model.named_modules()):
            if isinstance(module, LlamaRotaryEmbedding):
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, attr, LlamaRotaryEmbedding(config=model.config, device="cpu"))
    model.to(device)
    if artifact["model_type"] == "tiny":
        tokenizer = ByteTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = HFTokenizer(AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True))
    if cfg.cache_weights:
        for module in model.modules():
            if isinstance(module, QuantizedLoRALinear):
                module.cached_weight = module.dequantized_weight().to(module.A.dtype)
    return model, tokenizer, cfg, artifact["metadata"]


def config_from_args(args):
    cfg = Config(bits=args.bits, rank=args.rank, lora_alpha=args.lora_alpha, dropout=args.dropout,
                 qa_alpha=args.qa_alpha, loss_mode=args.loss_mode, learning_rate=args.lr,
                 epochs=args.epochs, init_epochs=args.init_epochs, init_fraction=args.init_fraction,
                 beta1=args.beta1, beta2=args.beta2, weight_decay=args.weight_decay,
                 lambda_update=args.lambda_update, max_length=args.max_length,
                 batch_size=args.batch_size, seed=args.seed, norm=args.norm,
                 cache_weights=not args.low_memory)
    cfg.validate()
    return cfg


def train_pipeline(args):
    output = Path(args.output)
    def prepare_output():
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output directory not empty: {output}; use a new output path")
        output.mkdir(parents=True, exist_ok=True)
    runtime.on_main(prepare_output)
    cfg = config_from_args(args)
    seed_all(cfg.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for smoke testing")
    records = load_records(args.data)
    train, validation, test = split_records(records, cfg.seed)
    def save_splits():
        for name, rows in (("train", train), ("validation", validation), ("test", test)):
            dump_jsonl(output / "splits" / f"{name}.jsonl", rows)
        dump_json(output / "config.json", cfg.to_dict())
    runtime.on_main(save_splits)
    print(f"[rank{runtime.rank()}] device={device} world_size={runtime.world_size()} "
          f"samples={len(records)} train/val/test={len(train)}/{len(validation)}/{len(test)} "
          f"global_batch_size={cfg.batch_size * runtime.world_size()}", flush=True)
    if cfg.loss_mode == "literal":
        warnings.warn("Paper-literal fixed Delta/detached lambda: QALoss penalty has ZERO LoRA gradient.")
    else:
        print(f"[rank{runtime.rank()}] QALoss loss_mode=residual qa_alpha={cfg.qa_alpha}: "
              "dynamic quantization residual participates in LoRA backpropagation; "
              "this modifies the paper's fixed-Delta formula and retains reference weights.", flush=True)
    model, tokenizer = load_backbone(args, cfg, device)
    install_loqdc(model, cfg)
    model.to(device)
    scores, history = fit(model, train, tokenizer, cfg, device, output)
    runtime.on_main(lambda: finalize_training(model, tokenizer, cfg, args, device, output,
                                             scores, records, validation, test))
    return output


def finalize_training(model, tokenizer, cfg, args, device, output, scores, records, validation, test):
    before = sum(p.numel() for p in model.parameters())
    prune = select_layers(model, scores, lambda: evaluate(model, validation, tokenizer, cfg, device),
                          args.max_accuracy_drop, args.remove_layers)
    after = sum(p.numel() for p in model.parameters())
    test_metrics = evaluate(model, test, tokenizer, cfg, device)
    software = {}
    for package in ("torch", "transformers", "numpy"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = None
    metadata = dict(pruning=prune, final_layer_scores=scores, parameters_before=before,
                    parameters_after=after, parameter_reduction=1 - after / before,
                    original_backbone=args.model or "offline-tiny", revision=args.revision,
                    software=software, synthetic=all(r.get("synthetic", False) for r in records),
                    test=test_metrics, loss_mode=cfg.loss_mode, world_size=runtime.world_size(),
                    global_batch_size=cfg.batch_size * runtime.world_size())
    save_artifact(model, tokenizer, cfg, output / "checkpoint", metadata)
    dump_json(output / "report.json", metadata)
    if args.export_hf:
        if isinstance(model, TinyClassifier):
            raise ValueError("--export-hf requires a CodeLlama backbone")
        merge_adapters(model)
        model.save_pretrained(output / "merged_hf", safe_serialization=True)
        tokenizer.save_pretrained(output / "merged_hf")
    print(json.dumps(dict(kept_layers=prune["selected"]["kept_original_indices"], test=test_metrics,
                          checkpoint=str(output / "checkpoint")), ensure_ascii=False, indent=2))
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("train", "demo"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", default="outputs/demo" if name == "demo" else "outputs/train")
        sub.add_argument("--device", default="cpu" if name == "demo" else "cuda")
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--bits", type=int, default=4)
        sub.add_argument("--rank", type=int, default=8)
        sub.add_argument("--lora-alpha", type=float, default=32)
        sub.add_argument("--dropout", type=float, default=0.05)
        sub.add_argument("--qa-alpha", type=float, default=0.01)
        sub.add_argument("--loss-mode", choices=("literal", "residual"), default="residual",
                         help="residual: differentiable post-LoRA quantization error (default); "
                              "literal: fixed error from the paper, with no penalty gradient")
        sub.add_argument("--lambda-update", choices=("epoch", "step"), default="epoch")
        sub.add_argument("--norm", choices=("fro", "spectral"), default="fro")
        sub.add_argument("--lr", type=float, default=1e-4)
        sub.add_argument("--epochs", type=int, default=3)
        sub.add_argument("--init-epochs", type=int, default=1)
        sub.add_argument("--init-fraction", type=float, default=0.1)
        sub.add_argument("--beta1", type=float, default=0.85)
        sub.add_argument("--beta2", type=float, default=0.85)
        sub.add_argument("--weight-decay", type=float, default=0.01)
        sub.add_argument("--max-length", type=int, default=128 if name == "demo" else 512)
        sub.add_argument("--batch-size", type=int, default=1)
        sub.add_argument("--tiny", action="store_true", default=name == "demo")
        sub.add_argument("--tiny-layers", type=int, default=4)
        sub.add_argument("--model", default="")
        sub.add_argument("--revision", default="main")
        sub.add_argument("--data", required=name == "train")
        sub.add_argument("--remove-layers", type=int, default=None,
                         help="fixed number to remove (paper threshold=23); omit for validation search")
        sub.add_argument("--max-accuracy-drop", type=float, default=0.01,
                         help="maximum ABSOLUTE validation accuracy drop")
        sub.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
        sub.add_argument("--gradient-checkpointing", action="store_true")
        sub.add_argument("--low-memory", action="store_true", help="dequantize on demand instead of caching")
        sub.add_argument("--export-hf", action="store_true", help="also export floating merged HF model")
    ev = commands.add_parser("evaluate")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--device", default="cpu")
    pred = commands.add_parser("predict")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--code-file", required=True, help="read-only classification; input code is NEVER executed")
    pred.add_argument("--device", default="cpu")
    return p


def main():
    args = parser().parse_args()
    try:
        args.device = str(runtime.initialize(args.device))
        if runtime.active() and args.command != "train":
            raise ValueError("torchrun is supported for train only; run other commands with python")
        execute_command(args)
    finally:
        runtime.cleanup()


def execute_command(args):
    if args.command in ("train", "demo"):
        if args.command == "demo":
            demo_input = Path(args.output).parent / (Path(args.output).name + "_input.jsonl")
            if demo_input.exists():
                raise FileExistsError(f"demo input already exists: {demo_input}; choose a new --output")
            dump_jsonl(demo_input, synthetic_records())
            args.data = str(demo_input)
            torch.set_num_threads(2)
        output = train_pipeline(args)
        if args.command == "demo":
            model, tokenizer, cfg, _ = load_artifact(output / "checkpoint", torch.device(args.device))
            metrics = evaluate(model, load_records(output / "splits" / "test.jsonl"), tokenizer, cfg,
                               torch.device(args.device))
            dump_json(output / "reload_test_metrics.json", metrics)
            print("Offline checkpoint reload and evaluation succeeded. Synthetic accuracy is NOT paper accuracy.")
    elif args.command == "evaluate":
        device = torch.device(args.device)
        model, tokenizer, cfg, _ = load_artifact(args.checkpoint, device)
        print(json.dumps(evaluate(model, load_records(args.data), tokenizer, cfg, device), indent=2))
    else:
        device = torch.device(args.device)
        model, tokenizer, cfg, _ = load_artifact(args.checkpoint, device)
        tokens = tokenizer.encode(Path(args.code_file).read_text(encoding="utf-8"), cfg.max_length)
        ids = torch.tensor([tokens], device=device)
        model.eval()
        with torch.no_grad():
            probabilities = logits_of(model, ids, torch.ones_like(ids)).float().softmax(-1)[0]
        print(json.dumps(dict(prediction=probabilities.argmax().item(), malicious_probability=probabilities[1].item()),
                         indent=2))


if __name__ == "__main__":
    main()
