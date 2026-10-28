"""Fixed inference storage reuse and opt-in CUDA accounting (sizes only)."""

import json
import os
from pathlib import Path
import time


class _FixedWeightNorm:
    def __init__(self, torch, original):
        self.torch = torch
        self.original = original
        self.name = original.name
        self.key = None
        self.weight = None
        self.storage_refs = None

    def parameters(self, module):
        return (getattr(module, self.name + "_g"), getattr(module, self.name + "_v"))

    def signature(self, module):
        return tuple((id(p), None if p.is_inference() else p._version,
                      p.untyped_storage()._cdata, p.dtype, p.device)
                     for p in self.parameters(module))

    def move(self, module, device):
        with self.torch.no_grad():
            for parameter in self.parameters(module):
                parameter.data = parameter.data.to(device)
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.to(device)

    def __call__(self, module, args):
        parameters = self.parameters(module)
        target = args[0].device if args and isinstance(args[0], self.torch.Tensor) else parameters[0].device
        if module.training or self.torch.is_grad_enabled() or any(p.grad is not None for p in parameters):
            self.key = self.weight = self.storage_refs = None
            self.move(module, target)
            self.original(module, args)
            return
        if self.key == self.signature(module) and self.weight is not None and self.weight.device == target:
            setattr(module, self.name, self.weight)
            return
        with self.torch.no_grad():
            self.move(module, target)
            self.original(module, args)
            self.weight = getattr(module, self.name)
            self.move(module, "cpu")
        self.key = self.signature(module)
        # Retain CPU storage identities to detect .half().float() even when the
        # allocator reuses addresses and Parameter version counters do not move.
        self.storage_refs = tuple(p.untyped_storage() for p in parameters)


def cache_decoder_weight_norm(torch, tts_model):
    """Reuse fixed effective decoder weights; original checkpoint keys stay CPU."""
    decoder = getattr(getattr(tts_model, "audio_vae", None), "decoder", None)
    if not isinstance(decoder, torch.nn.Module) or decoder.training:
        return {"enabled": False, "reason": "no_eval_decoder"}
    candidates = []
    ids = set()
    encoder = getattr(getattr(tts_model, "audio_vae", None), "encoder", None)
    encoder_ids = {id(p) for p in encoder.parameters()} if isinstance(encoder, torch.nn.Module) else set()
    for module in decoder.modules():
        for key, hook in module._forward_pre_hooks.items():
            if type(hook).__name__ == "WeightNorm" and type(hook).__module__ == "torch.nn.utils.weight_norm":
                params = (getattr(module, hook.name + "_g"), getattr(module, hook.name + "_v"))
                if any(id(p) in ids or id(p) in encoder_ids or p.grad is not None or p.is_inference()
                       for p in params):
                    return {"enabled": False, "reason": "shared_or_unversioned_or_training_norm_parameters"}
                ids.update(id(p) for p in params)
                candidates.append((module, key, hook))
    if not candidates:
        existing = any(isinstance(hook, _FixedWeightNorm) for module in decoder.modules()
                       for hook in module._forward_pre_hooks.values())
        return {"enabled": existing, "reason": "already_cached" if existing else "no_legacy_weight_norm"}
    device = getattr(candidates[0][0], candidates[0][2].name + "_v").device
    if device.type != "cuda":
        return {"enabled": False, "reason": "no_cuda_decoder"}
    cpu_bytes = 0
    with torch.no_grad():
        for module, key, original in candidates:
            replacement = _FixedWeightNorm(torch, original)
            cpu_bytes += sum(p.numel() * p.element_size() for p in replacement.parameters(module))
            replacement(module, ())
            module._forward_pre_hooks[key] = replacement
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return {"enabled": True, "modules_cached": len(candidates),
            "original_parameter_bytes_kept_on_cpu": cpu_bytes}


def enable_prompt_encoder_offload(torch, tts_model):
    """Offload the prompt-only AudioVAE encoder, reloading transparently if used.

    Legacy weight_norm keeps normalized weights as plain tensor attributes;
    Module.cpu() alone does not release those cached CUDA tensors.
    """
    vae = getattr(tts_model, "audio_vae", None)
    encoder, decoder = getattr(vae, "encoder", None), getattr(vae, "decoder", None)
    if not isinstance(encoder, torch.nn.Module) or not isinstance(decoder, torch.nn.Module):
        return {"enabled": False, "reason": "no_separate_vae_encoder_decoder"}
    if encoder.training:
        return {"enabled": False, "reason": "training_encoder"}
    if getattr(encoder, "_openva_offload_enabled", False):
        return {"enabled": True, "already_enabled": True}
    source = list(encoder.parameters()) + list(encoder.buffers())
    other = list(decoder.parameters()) + list(decoder.buffers())
    if {id(t) for t in source} & {id(t) for t in other}:
        return {"enabled": False, "reason": "shared_encoder_decoder_state"}
    device = next((t.device for t in source if t.is_cuda), None)
    if device is None:
        return {"enabled": False, "reason": "no_cuda_encoder"}

    def move(module, target):
        module.to(target)
        for child in module.modules():
            for name, tensor in tuple(vars(child).items()):
                if isinstance(tensor, torch.Tensor):
                    setattr(child, name, tensor.to(target))

    def before(module, args):
        if args and isinstance(args[0], torch.Tensor):
            move(module, args[0].device)

    def after(module, args, output):
        if not module.training and not torch.is_grad_enabled():
            move(module, "cpu")
            if args and isinstance(args[0], torch.Tensor) and args[0].is_cuda:
                with torch.cuda.device(args[0].device):
                    torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    before_bytes = torch.cuda.memory_allocated(device)
    inventory = cuda_storage_inventory(torch, {"encoder": encoder})
    with torch.no_grad():
        move(encoder, "cpu")
    torch.cuda.synchronize(device)
    encoder.register_forward_pre_hook(before)
    encoder.register_forward_hook(after, always_call=True)
    encoder._openva_offload_enabled = True
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return {"enabled": True, "encoder_cuda_storage_bytes": inventory["unique_cuda_storage_bytes"],
            "allocated_freed_bytes": before_bytes - torch.cuda.memory_allocated(device)}


def prepare_motion_inference_memory(torch, model):
    """Prepare the fixed eval-only motion worker before any CUDA graph capture.

    These modules are absent from the production forward arithmetic. Keep their
    weights on CPU for checkpoint compatibility. Only share position buffers
    after comparing all values; no precision or context length is changed.
    """
    if model.training:
        raise ValueError("motion memory preparation requires eval mode")
    if getattr(model, "cuda_graph_enabled", False):
        raise ValueError("prepare motion memory before CUDA graph capture")
    unused = [module for block in model.blocks
              for module in (block.cross_attn_rope, block.norm2)]
    unused.extend((model.diffusion_head.past_proj, model.diffusion_head.anchor_proj))
    moved = sum(t.numel() * t.element_size() for module in unused
                for tensors in (module.parameters(), module.buffers()) for t in tensors if t.is_cuda)
    for module in unused:
        module.cpu()
    pools = {}
    shared_bytes = 0
    shared_buffers = 0
    for block in model.blocks:
        for module in (block.self_attn_rope.rope, block.cross_attn_anchor.rope, block.self_attn_pos.pe):
            for field in ("cos_angles", "sin_angles", "pe"):
                buffer = module._buffers.get(field)
                if buffer is None or not buffer.is_cuda:
                    continue
                key = (field, tuple(buffer.shape), buffer.dtype, buffer.device)
                candidates = pools.setdefault(key, [])
                reference = next((value for value in candidates
                                  if value is buffer or torch.equal(value, buffer)), None)
                if reference is None:
                    candidates.append(buffer)
                elif reference is not buffer:
                    shared_bytes += buffer.numel() * buffer.element_size()
                    shared_buffers += 1
                    setattr(module, field, reference)
    return {"unused_cuda_tensor_bytes_offloaded": moved,
            "identical_position_buffer_bytes_shared": shared_bytes,
            "position_buffers_shared": shared_buffers}


def place_motion_for_inference(torch, model, device):
    """Place a CPU-loaded fixed model without allocating inactive CUDA weights.

    Sharing after a normal .cuda() leaves inactive split blocks behind. Build
    the same aliases on CPU, then memoize transfers so shared buffers stay shared.
    """
    if model.training or getattr(model, "cuda_graph_enabled", False):
        raise ValueError("place an eval motion model before CUDA graph capture")
    tensors = list(model.parameters()) + list(model.buffers())
    if any(t.device.type != "cpu" for t in tensors):
        raise ValueError("selective motion placement requires a CPU-loaded model")
    unused = [module for block in model.blocks
              for module in (block.cross_attn_rope, block.norm2)]
    unused.extend((model.diffusion_head.past_proj, model.diffusion_head.anchor_proj))
    inactive = {id(t) for module in unused
                for values in (module.parameters(), module.buffers()) for t in values}
    skipped_bytes = sum(t.numel() * t.element_size() for t in tensors if id(t) in inactive)
    pools = {}
    shared_bytes = 0
    for block in model.blocks:
        for module in (block.self_attn_rope.rope, block.cross_attn_anchor.rope, block.self_attn_pos.pe):
            for field in ("cos_angles", "sin_angles", "pe"):
                value = module._buffers.get(field)
                if value is None:
                    continue
                key = (field, tuple(value.shape), value.dtype)
                candidates = pools.setdefault(key, [])
                reference = next((p for p in candidates if p is value or torch.equal(p, value)), None)
                if reference is None:
                    candidates.append(value)
                elif reference is not value:
                    shared_bytes += value.numel() * value.element_size()
                    setattr(module, field, reference)
    moved = {}
    def transfer(tensor):
        if id(tensor) in inactive:
            return tensor
        key = id(tensor)
        if key not in moved:
            moved[key] = tensor.to(device)
        return moved[key]
    model._apply(transfer)
    return {"inactive_tensor_bytes_kept_on_cpu": skipped_bytes,
            "identical_position_buffer_bytes_shared": shared_bytes}


def cuda_storage_inventory(torch, modules, tensors=None):
    seen = set()
    entries = {}

    def add(label, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            return
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            entries[label] = storage.nbytes()

    for name, module in modules.items():
        for key, tensor in module.named_parameters():
            add(f"{name}.parameter.{key}", tensor)
        for key, tensor in module.named_buffers():
            add(f"{name}.buffer.{key}", tensor)
        # Spectral-normalized weights and our inference caches are ordinary
        # tensor attributes and do not appear in named_parameters/buffers.
        for prefix, child in module.named_modules():
            for key, value in vars(child).items():
                if isinstance(value, torch.Tensor):
                    add(f"{name}.attribute.{prefix}.{key}", value)

    containers = set()
    def visit(label, value):
        if isinstance(value, torch.Tensor):
            add(label, value)
        elif isinstance(value, (dict, tuple, list)) and id(value) not in containers:
            containers.add(id(value))
            items = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in items:
                visit(f"{label}.{key}", item)

    visit("runtime", tensors or {})
    return {"unique_cuda_storage_bytes": sum(entries.values()), "storage_bytes": entries}


class MemoryRecorder:
    def __init__(self, role, torch, device):
        directory = os.getenv("OPENVA_MEMORY_PROFILE_DIR", "").strip()
        self.enabled = bool(directory)
        self.torch = torch
        self.device = torch.device(device)
        self.path = None
        if self.enabled:
            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=True)
            self.path = folder / f"{role}-{os.getpid()}-{time.time_ns()}.jsonl"
        self.role = role

    def record(self, stage, modules, tensors=None):
        if not self.enabled:
            return
        torch = self.torch
        if self.device.type != "cuda":
            return
        torch.cuda.synchronize(self.device)
        inventory = cuda_storage_inventory(torch, modules, tensors)
        stats = torch.cuda.memory_stats(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        result = {
            "role": self.role, "pid": os.getpid(), "stage": stage,
            "time_ns": time.time_ns(), "device": str(self.device),
            "allocated_bytes": allocated,
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device),
            "inactive_split_bytes": stats.get("inactive_split_bytes.all.current", 0),
            "allocation_retries": stats.get("num_alloc_retries", 0),
            **inventory,
            "allocated_minus_labelled_storage_bytes": allocated - inventory["unique_cuda_storage_bytes"],
            "allocator_backend": torch.cuda.get_allocator_backend(),
        }
        pools = {}
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        segments = torch.cuda.memory_snapshot() if result["allocator_backend"] == "native" else []
        for segment in segments:
            if segment["device"] != index:
                continue
            key = str(tuple(segment.get("segment_pool_id", (0, 0))))
            item = pools.setdefault(key, {"reserved_bytes": 0, "allocated_bytes": 0, "active_bytes": 0})
            for field, source in (("reserved_bytes", "total_size"), ("allocated_bytes", "allocated_size"),
                                  ("active_bytes", "active_size")):
                item[field] += segment.get(source, 0)
        result["allocator_pools"] = pools
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result
