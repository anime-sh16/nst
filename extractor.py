import torch
import torch.nn as nn
import torchvision.models as models
from typing import cast

class VGG19FeatureExtractor(nn.Module):
    """VGG19 feature extractor using forward hooks to capture intermediate activations.

    MaxPool layers are replaced with AvgPool as recommended in Gatys et al.
    it produces smoother gradients during optimization.
    layer_names maps a friendly name to the index in vgg19.features.
    """
    
    def __init__(self, layer_names = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vgg = cast(nn.Sequential, models.vgg19(weights=models.VGG19_Weights.DEFAULT).features)
        self.vgg.requires_grad_(False)
        self.vgg.eval()
        
        for i, layer in enumerate(self.vgg):
            if isinstance(layer, nn.MaxPool2d):
                # VGG's default max pool is kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                self.vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
        self.layer_names = layer_names
        self.layers = {}
        self.handles = []
        
        self._register_hooks()
    
    def show_layers(self):
        for idx, layer in enumerate(self.vgg):
            print(f"Index {idx}: {layer}")
            if isinstance(layer, nn.modules.pooling.MaxPool2d):
                print("--"*30)
        
    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
    
    def _register_hooks(self):
        for name, idx in self.layer_names.items():
            layer = self.vgg[idx]
            handle = layer.register_forward_hook(
                    lambda module, input, output, name=name: self.layers.update({name: output})
                )
            self.handles.append(handle)
    
    def forward(self, x):
        self.layers.clear() 
        self.vgg(x)          
        return self.layers.copy()

    def __del__(self):
        self._remove_hooks()