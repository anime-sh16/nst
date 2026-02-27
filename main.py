import argparse
import json
import torch
from torchvision.transforms import v2
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from extractor import VGG19FeatureExtractor
from loss import calculate_style_total_loss, calculate_content_loss
from config import STYLE_KEYS, STYLE_LOSS_WEIGHTS, CONTENT_KEY, LAYER_INDICES, ALPHA, BETA, MAX_STEPS
from utils import load_image, denormalize


def run_style_transfer(
    content_path: Path,
    style_path: Path,
    output_path: Path,
    alpha: float = 1.0,
    beta: float = 1e8,
    steps: int = 1000,
):
    """Run neural style transfer and save the result to output_path.

    alpha and beta weight the content and style losses respectively.
    Their ratio is what matters, higher beta/alpha means more stylized
    at the cost of content structure.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    content_tensor = load_image(content_path).to(device)
    style_tensor = load_image(style_path).to(device)
    
    x_size = content_tensor.shape[2]
    y_size = content_tensor.shape[3]
    
    vgg = VGG19FeatureExtractor(layer_names=LAYER_INDICES).to(device)

    with torch.no_grad():
        content_features = vgg(content_tensor)
        style_features = vgg(style_tensor)

    # paper allows initializing from content image too, but noise works fine with LBFGS
    gen_tensor = torch.randn(1, 3, x_size, y_size, device=device, requires_grad=True)
    # LBFGS requires a closure that recomputes the loss on every call,
    # so gen_tensor is optimized in-place via the closure below.
    optimizer = optim.LBFGS([gen_tensor], line_search_fn="strong_wolfe", max_iter=steps)

    step = 0

    def closure():
        nonlocal step
        optimizer.zero_grad()

        gen_features = vgg(gen_tensor)

        content_loss = calculate_content_loss(
            content_features[CONTENT_KEY], gen_features[CONTENT_KEY]
        )
        style_loss = calculate_style_total_loss(
            STYLE_LOSS_WEIGHTS,
            [style_features[k] for k in STYLE_KEYS],
            [gen_features[k] for k in STYLE_KEYS],
        )

        total_loss = alpha * content_loss + beta * style_loss
        total_loss.backward()

        step += 1
        if step % 50 == 0:
            print(
                f"Step {step:4d}/{steps} | "
                f"total: {total_loss.item():.4e} | "
                f"content: {(alpha * content_loss).item():.4e} | "
                f"style: {(beta * style_loss).item():.4e}"
            )

        return total_loss

    print("Starting optimization...")
    optimizer.step(closure)

    result = denormalize(gen_tensor.detach().squeeze(0), device).cpu()
    result_img = v2.ToPILImage()(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"Saved to {output_path}")

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "output": str(output_path),
        "content": str(content_path),
        "style": str(style_path),
        "alpha": alpha,
        "beta": beta,
        "steps": steps,
        "style_keys": STYLE_KEYS,
        "style_loss_weights": STYLE_LOSS_WEIGHTS,
        "content_key": CONTENT_KEY,
    }
    log_path = output_path.parent / "runs.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer (Gatys et al., 2015)")
    parser.add_argument("--content", required=True, type=Path, help="Path to content image")
    parser.add_argument("--style", required=True, type=Path, help="Path to style image")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: output/<content>_x_<style>.jpg)")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Optimization steps")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Content loss weight")
    parser.add_argument("--beta", type=float, default=BETA, help="Style loss weight")
    args = parser.parse_args()

    if args.output is None:
        content_stem = args.content.stem
        style_stem = args.style.stem
        base = Path("outputs") / f"{content_stem}_x_{style_stem}"
        args.output = base.with_suffix(".jpg")
        if args.output.exists():
            n = 2
            while (candidate := base.with_name(f"{base.name}_{n}").with_suffix(".jpg")).exists():
                n += 1
            args.output = candidate

    run_style_transfer(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        alpha=args.alpha,
        beta=args.beta,
        steps=args.steps,
    )


if __name__ == "__main__":
    main()