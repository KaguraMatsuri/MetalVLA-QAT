"""Official Qwen with the retained trained decoder MLP weights."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class EncoderConfig:
    base_model: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    dtype: str = "float32"
    attention: str = "eager"
    image_size: int = 192
    backbone_path: str | None = None
    backbone_sha256: str | None = None


def prepare_messages(raw_rgb, instructions, image_size=192):
    import torch
    from PIL import Image
    from ..rgb import preprocess_rgb
    raw = np.asarray(raw_rgb)
    if raw.ndim != 5 or raw.shape[1] != 3 or raw.shape[-1] != 3 or raw.dtype != np.uint8:
        raise ValueError("Expected CPU uint8 RGB [batch,3,height,width,3]")
    if isinstance(instructions, str):
        instructions = [instructions] * len(raw)
    if len(instructions) != len(raw):
        raise ValueError("Instructions and observations must align")
    # Match the old letterbox and uint8 rounding exactly, entirely on CPU.
    images = preprocess_rgb(torch.from_numpy(raw.copy()), image_size)
    pixels = images.permute(0, 1, 3, 4, 2).mul(255).round().to(torch.uint8).numpy()
    return messages_from_pixels(pixels, instructions)


def messages_from_pixels(pixels, instructions):
    from PIL import Image
    messages = []
    for text, frames in zip(instructions, pixels, strict=True):
        content = []
        for name, frame in zip(("overhead", "wrist", "side"), frames, strict=True):
            content.extend(({"type": "text", "text": f"{name} camera:"},
                            {"type": "image", "image": Image.fromarray(frame)}))
        content.append({"type": "text", "text": text})
        messages.append([{"role": "user", "content": content}])
    return messages


class OfficialQwenEncoder:
    def __init__(self, config=EncoderConfig(), device="mps"):
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor
        if not config.backbone_path or not config.backbone_sha256:
            raise ValueError({"backbone_path": config.backbone_path, "backbone_sha256": config.backbone_sha256})
        if config.dtype not in ("float32", "float16", "bfloat16") or config.attention not in ("eager", "sdpa"):
            raise ValueError("Unsupported official runtime profile")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        upstream = AutoConfig.from_pretrained(config.base_model, revision=config.revision)
        if upstream.model_type != "qwen3_5":
            raise ValueError("This encoder currently validates Qwen3.5 multimodal models only")
        self.hidden_size = upstream.text_config.hidden_size
        self.config, self.device = config, device
        self.processor = AutoProcessor.from_pretrained(config.base_model, revision=config.revision)
        self.processor.tokenizer.padding_side = "right"
        self.processor.image_processor.size = {"shortest_edge": config.image_size**2,
                                               "longest_edge": config.image_size**2}
        self.model = AutoModel.from_pretrained(config.base_model, revision=config.revision,
                                              dtype=getattr(torch, config.dtype),
                                              attn_implementation=config.attention).to(device)
        if config.backbone_path:
            from safetensors.torch import load_file
            from .data import file_hash
            if file_hash(config.backbone_path) != config.backbone_sha256:
                raise ValueError({"backbone_sha256": config.backbone_sha256})
            weights = load_file(config.backbone_path)
            layers = self.model.language_model.layers
            expected = {f"language_model.layers.{index}.mlp.{name}"
                        for index in range(len(layers)-2, len(layers))
                        for name in layers[index].mlp.state_dict()}
            if set(weights) != expected:
                raise ValueError({"missing": sorted(expected-set(weights)), "unexpected": sorted(set(weights)-expected)})
            state = self.model.state_dict()
            state.update(weights)
            self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.timings = []

    def prepare(self, raw_rgb, instructions):
        started = time.perf_counter()
        messages = prepare_messages(raw_rgb, instructions, self.config.image_size)
        tensors = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
        )
        return tensors, time.perf_counter() - started

    def encode_prepared(self, prepared):
        import torch
        tensors, preprocessing_s = prepared
        started = time.perf_counter()
        with torch.inference_mode():
            tensors = tensors.to(self.device)
            output = self.model(**tensors, use_cache=False, return_dict=True)
            rows = torch.arange(len(tensors["attention_mask"]), device=self.device)
            final = tensors["attention_mask"].sum(1) - 1
            features = output.last_hidden_state[rows, final].float().cpu().numpy()
        if features.shape != (len(rows), self.hidden_size) or not np.isfinite(features).all():
            raise FloatingPointError("Invalid official Qwen feature output")
        self.timings.append(dict(preprocessing_s=preprocessing_s,
                                 gpu_and_feature_transfer_s=time.perf_counter()-started,
                                 batch=len(features), image_transfers_to_mps=1,
                                 image_transfers_from_mps=0))
        return features

    def encode(self, raw_rgb, instructions):
        return self.encode_prepared(self.prepare(raw_rgb, instructions))

    def signature(self):
        config = asdict(self.config)
        config.pop("backbone_path")
        value = dict(encoder=config, hidden_size=self.hidden_size,
                     preprocessing="letterbox_bilinear_antialias_round_uint8_v1",
                     camera_order=["overhead", "wrist", "side"], readout="last_valid_token")
        return dict(**value, sha256=hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


def assert_base_matches(policy, encoder):
    c, e = policy.config, encoder.config
    if (c.base_model, c.base_revision, c.feature_dim, c.image_size, c.backbone_sha256) != (
            e.base_model, e.revision, encoder.hidden_size, e.image_size, e.backbone_sha256):
        raise ValueError("Base model changed: rebuild/retrain the action head and invalidate the feature cache")
