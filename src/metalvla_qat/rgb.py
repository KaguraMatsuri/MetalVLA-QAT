"""Shared letterboxing without importing a policy, TorchAO or Core ML."""


def preprocess_rgb(images, image_size=224):
    import torch
    from torch.nn import functional as F
    if images.ndim != 5 or images.shape[1] != 3 or images.shape[-1] != 3 or images.dtype != torch.uint8:
        raise ValueError("Expected uint8 RGB [B,3,H,W,3] in overhead/wrist/side order")
    batch, cams, height, width, _ = images.shape
    if min(batch, height, width) < 1:
        raise ValueError("Empty RGB input")
    if type(image_size) is not int or image_size < 64 or image_size % 32:
        raise ValueError("image_size must be at least 64 and divisible by 32")
    scale = image_size / max(height, width)
    h, w = max(1, round(height * scale)), max(1, round(width * scale))
    values = images.permute(0, 1, 4, 2, 3).reshape(batch * cams, 3, height, width).float() / 255
    values = F.interpolate(values, size=(h, w), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
    left, top = (image_size-w)//2, (image_size-h)//2
    values = F.pad(values, (left, image_size-w-left, top, image_size-h-top))
    return values.reshape(batch, cams, 3, image_size, image_size)
