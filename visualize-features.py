import torch
from matplotlib import pyplot as plt
from pathlib import Path

from extractor import VGG19FeatureExtractor
from loss import get_gram_matrix
from config import LAYER_INDICES
from utils import load_image

def plot_feature_maps(features_tensor: torch.Tensor, output_path: Path, num_channels: int = 10):
    """Plots the first `num_channels` of a feature map and its Gram matrix."""
    # Ensure we don't try to plot more channels than exist
    num_channels = min(num_channels, features_tensor.size(1))
    
    # Isolate the requested channels and move to CPU
    selected_channels = features_tensor[:, :num_channels, :, :]
    squeezed_channels = selected_channels.squeeze(0).cpu().detach()
    
    fig, axes = plt.subplots(4, 3, figsize=(6, 12))
    axes = axes.flatten()
    
    # Plot individual feature channels
    for i in range(num_channels):
        axes[i].imshow(squeezed_channels[i].numpy(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"Channel {i+1}")
        
    # Calculate and plot the Gram matrix on the 11th subplot (index 10)
    gram = get_gram_matrix(selected_channels)
    gram = gram.squeeze(0).cpu().detach()
    
    im = axes[10].imshow(gram.numpy(), cmap='magma')
    fig.colorbar(im, ax=axes[10], fraction=0.046, pad=0.04)
    
    ticks = range(num_channels)
    labels = [str(i+1) for i in range(num_channels)]
    
    axes[10].set_xticks(ticks)
    axes[10].set_yticks(ticks)
    axes[10].set_xticklabels(labels, fontsize=8)
    axes[10].set_yticklabels(labels, fontsize=8)
    axes[10].set_title("Gram Matrix")
    
    # Hide the 12th subplot (index 11) as it is unused
    axes[11].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()

def main():
    script_dir = Path(__file__).parent.resolve()
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dir = script_dir / "inputs"
    output_base_dir = script_dir / "features"
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return

    # Safely gather only image files
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions]
    
    print(f"Number of images to process: {len(image_paths)}")
    if not image_paths:
        return

    vgg_obj = VGG19FeatureExtractor(layer_names=LAYER_INDICES).to(device)

    # Disable gradient calculation for purely visual inference
    with torch.no_grad():
        for image_path in image_paths:
            print(f"Extracting features for: {image_path.name}")
            output_dir = output_base_dir / image_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)
            
            image_tensor = load_image(image_path).to(device)
            features = vgg_obj(image_tensor)

            for layer_name, feature in features.items():
                out_file = output_dir / f"{layer_name}.jpg"
                plot_feature_maps(feature, out_file)

if __name__ == "__main__":
    main()