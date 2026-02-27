import torch
from torchvision.transforms import v2

# Style layers and per-layer weights as used in Gatys et al.
STYLE_KEYS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
STYLE_LOSS_WEIGHTS = [0.25, 0.25, 0.25, 0.15, 0.25]
CONTENT_KEY = "conv4_2"

ALPHA = 1.0
BETA = 1e8
MAX_STEPS = 1000

# indices into vgg19.features sequential (0-based)
# the indices are actually for the output after the relu layer of the corresponding conv layer
# the keys use the names to match the names used in Gatys et al. Thats all.
LAYER_INDICES = {"conv1_1": 1, "conv2_1": 6, "conv3_1": 11, "conv4_1": 20, "conv4_2": 22, "conv5_1": 29}

IMG_SIZE = 256  # VGG19 expected input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(IMG_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])