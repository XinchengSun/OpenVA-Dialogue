"""Optional CPU text lookup; preserve the original table, dtype and output device."""

import torch
from torch import nn


class _CPUTextEmbedding(nn.Embedding):
    def _move_weight(self, device):
        with torch.no_grad():
            self.weight.data = self.weight.data.to(device)
            if self.weight.grad is not None:
                self.weight.grad.data = self.weight.grad.data.to(device)

    def forward(self, input):
        target = self._openva_device_marker.device
        # Preserve native autograd/training and incompatible-device behavior.
        fallback = (self.training or torch.is_grad_enabled() or self.weight.grad is not None or not input.is_cuda
                    or (input.is_cuda and input.device != target))
        if fallback:
            if self.weight.device != target:
                self._move_weight(target)
            return super().forward(input)
        if self.weight.device.type != "cpu":
            self._move_weight("cpu")
        return super().forward(input.to("cpu")).to(target)


def offload_text_embedding(tts_model):
    """Move the untied eval text table to CPU; move only selected rows to CUDA.

    Official VoxCPM2 calls this table before LM prefill, outside compiled step
    decoding. A zero-byte nonpersistent buffer tracks .to()/dtype conversions
    without adding checkpoint keys or retaining a GPU copy of the table.
    """
    if not isinstance(tts_model, nn.Module):
        return {"enabled": False, "reason": "requires_module_owner"}
    embedding = getattr(getattr(tts_model, "base_lm", None), "embed_tokens", None)
    if isinstance(embedding, _CPUTextEmbedding):
        return {"enabled": True, "reason": "already_offloaded"}
    if type(embedding) is not nn.Embedding or embedding.training:
        return {"enabled": False, "reason": "requires_standard_eval_embedding"}
    if embedding.max_norm is not None or embedding._compiled_call_impl is not None:
        return {"enabled": False, "reason": "mutable_or_compiled_embedding"}
    weight = embedding.weight
    if not weight.is_cuda or weight.grad is not None or weight.is_inference():
        return {"enabled": False, "reason": "requires_cuda_inference_weight"}
    aliases = [parameter for _, parameter in tts_model.named_parameters(remove_duplicate=False)
               if parameter.is_cuda and parameter.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()]
    if len(aliases) != 1:
        return {"enabled": False, "reason": "shared_embedding_storage"}
    device = weight.device
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    storage_bytes = weight.numel() * weight.element_size()
    embedding.__class__ = _CPUTextEmbedding
    embedding.register_buffer("_openva_device_marker", torch.empty(0, device=device), persistent=False)
    embedding._move_weight("cpu")
    torch.cuda.synchronize(device)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return {"enabled": True, "table_bytes_kept_on_cpu": storage_bytes,
            "allocated_freed_bytes": before - torch.cuda.memory_allocated(device),
            "dtype": str(weight.dtype), "shape": list(weight.shape)}
