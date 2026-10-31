"""Optional torchrun DDP runtime; single-process entrypoints remain supported."""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def initialize(device_name):
    device = torch.device(device_name)
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return device
    if device.type == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"rank {local_rank} has no accessible CUDA device")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                            timeout=timedelta(hours=2))
    return device


def on_main(operation):
    """Run one filesystem/evaluation operation, propagating failures to every rank."""
    if not active():
        return operation()
    error = [None]
    result = None
    if rank() == 0:
        try:
            result = operation()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(f"rank-0 operation failed: {error[0]}")
    return result


def reduce_totals(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if active():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def cleanup():
    if active():
        dist.destroy_process_group()
