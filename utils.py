import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2
from pathlib import Path
from config import preprocess, IMAGENET_MEAN, IMAGENET_STD


def load_image(path: Path) -> torch.Tensor:
    img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    return preprocess(img).unsqueeze(0)


def denormalize(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)