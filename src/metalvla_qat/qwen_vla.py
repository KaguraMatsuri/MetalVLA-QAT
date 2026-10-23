"""Official Qwen encoder used by decoder-MLP fine-tuning."""
import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor, Qwen3_5Model

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CAMERAS = ("overhead", "wrist", "side")

class QwenVLA(nn.Module):
    def __init__(self, vocabulary, base_model=BASE_MODEL, base_revision=BASE_REVISION, image_size=192, backbone_dtype="float32"):
        super().__init__()
        if image_size != 192:
            raise ValueError("This grasp trial expects the existing 192px cameras")
        self.words = {value: word for word, value in vocabulary.items()}
        if backbone_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError({"backbone_dtype": backbone_dtype})
        self.processor = AutoProcessor.from_pretrained(base_model, revision=base_revision)
        self.processor.tokenizer.padding_side = "right"
        self.processor.image_processor.size = {"shortest_edge": image_size**2, "longest_edge": image_size**2}
        self.backbone = Qwen3_5Model.from_pretrained(base_model, revision=base_revision,
            dtype=getattr(torch, backbone_dtype), attn_implementation="eager")
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def extract_features(
            self, images: torch.Tensor, token_ids: torch.Tensor,
        ) -> torch.Tensor:
            if images.ndim != 5 or tuple(images.shape[1:]) != (3, 3, 192, 192):
                raise ValueError(f"Expected [B,3,3,192,192] images, got {images.shape}")
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("Images must be finite RGB values in [0,1]")
            pixels = (
                images.detach().cpu().permute(0, 1, 3, 4, 2).mul(255)
                .round().to(torch.uint8).numpy()
            )
            messages = []
            for row, frame in zip(token_ids.detach().cpu().tolist(), pixels, strict=True):
                instruction = " ".join(self.words[token] for token in row if token)
                content = []
                for camera, picture in zip(CAMERAS, frame, strict=True):
                    content.extend((
                        {"type": "text", "text": f"{camera} camera:"},
                        {"type": "image", "image": Image.fromarray(picture)},
                    ))
                content.append({"type": "text", "text": instruction})
                messages.append([{"role": "user", "content": content}])
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True},
            ).to(next(self.backbone.parameters()).device)
            output = self.backbone(**inputs, use_cache=False, return_dict=True)
            # The final valid token can attend to all three images and the instruction.
            final_token = inputs["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(messages), device=final_token.device)
            features = output.last_hidden_state[rows, final_token]
            if not torch.isfinite(features).all():
                raise FloatingPointError("Qwen produced non-finite visual-language features")
            return features
