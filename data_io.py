"""JSONL data validation, deterministic stratified splits and token batching."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Dataset


class HFTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def encode(self, code, max_length):
        return self.tokenizer(code, truncation=True, max_length=max_length,
                              add_special_tokens=True)["input_ids"]

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)


def load_records(path) -> list[dict]:
    records, seen_ids, seen_code = [], set(), {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("code"), str) or not row["code"].strip():
                raise ValueError(f"{path}:{line_number}: non-empty string code required")
            if type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError(f"{path}:{line_number}: label must be integer 0 (benign) or 1 (malicious)")
            if row.get("language") not in ("python", "javascript"):
                raise ValueError(f"{path}:{line_number}: language must be python or javascript")
            code_hash = hashlib.sha256(row["code"].strip().encode("utf-8")).hexdigest()
            row = dict(row)
            row["id"] = str(row.get("id", code_hash))
            if row["id"] in seen_ids or code_hash in seen_code:
                raise ValueError(f"{path}:{line_number}: duplicate ID/code; deduplicate before splitting")
            seen_ids.add(row["id"])
            seen_code[code_hash] = row["label"]
            records.append(row)
    if not records:
        raise ValueError(f"empty dataset: {path}")
    return records


def split_records(records, seed=42) -> tuple[list[dict], list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in records:
        groups[(row["language"], row["label"])].append(row)
    rng = random.Random(seed)
    result = ([], [], [])
    for key in sorted(groups):
        group = groups[key][:]
        if len(group) < 10:
            raise ValueError(f"stratum {key} has {len(group)} samples; need at least 10 for a 7:1:2 split")
        rng.shuffle(group)
        n_val, n_test = max(1, round(len(group) * 0.1)), max(1, round(len(group) * 0.2))
        n_train = len(group) - n_val - n_test
        result[0].extend(group[:n_train])
        result[1].extend(group[n_train:n_train + n_val])
        result[2].extend(group[n_train + n_val:])
    for group in result:
        rng.shuffle(group)
    return result


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dump_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


class TokenDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.samples = [(tokenizer.encode(row["code"], max_length), row["label"]) for row in records]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        length = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), length), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        for i, (tokens, _) in enumerate(batch):
            ids[i, :len(tokens)] = torch.tensor(tokens)
            mask[i, :len(tokens)] = 1
        return ids, mask, labels


def loader(records, tokenizer, cfg, shuffle=False, distributed=False, epoch=0):
    dataset = TokenDataset(records, tokenizer, cfg.max_length)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=cfg.seed, drop_last=False)
        sampler.set_epoch(epoch)
    return DataLoader(dataset, batch_size=cfg.batch_size,
                      shuffle=shuffle if sampler is None else False, sampler=sampler,
                      collate_fn=Collator(tokenizer.pad_token_id),
                      generator=torch.Generator().manual_seed(cfg.seed))


def synthetic_records(n_per_stratum=20):
    """Safe, artificial labels for pipeline testing, NOT a detection benchmark."""
    rows = []
    for language in ("python", "javascript"):
        for label in (0, 1):
            for index in range(n_per_stratum):
                marker = "audit_fixture" if label else "ordinary_fixture"
                code = (f"# {marker} {index}\nvalue = {index}\nprint(value)\n" if language == "python"
                        else f"// {marker} {index}\nconst value = {index};\nconsole.log(value);\n")
                rows.append(dict(id=f"demo-{language}-{label}-{index}", code=code, label=label,
                                 language=language, synthetic=True))
    return rows
