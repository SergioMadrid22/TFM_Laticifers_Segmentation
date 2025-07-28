import argparse, yaml, os, torch, numpy as np, random, logging
from datasets import get_patch_dataloaders
from train import test_model
from skimage.morphology import binary_closing, remove_small_objects, skeletonize, disk
from skimage.measure import label
from utils import get_loss_function, reconstruct_from_patches
import matplotlib.pyplot as plt
from metrics import compute_metrics
from clDice.cldice_metric.cldice import clDice as compute_cldice
from PIL import Image

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained segmentation model")
    parser.add_argument('-c', '--config', required=True, help='Path to config YAML')
    parser.add_argument('-m', '--model', required=True, help='Path to trained model (e.g., best_model.pth)')
    parser.add_argument('-o', '--output', default='test_outputs', help='Directory to save visual outputs')
    parser.add_argument('--tta', action='store_true', help='Enable Test-Time Augmentation (TTA)')
    return parser.parse_args()

# ------------------ TTA Function -------------------

def apply_tta(model, patch_batch):
    """Apply test-time augmentation (flips) and average predictions."""
    flips = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),      # horizontal
        lambda x: torch.flip(x, dims=[-2]),      # vertical
        lambda x: torch.flip(x, dims=[-1, -2])   # both
    ]

    preds = []
    for flip_fn in flips:
        flipped_input = flip_fn(patch_batch)
        with torch.no_grad():
            pred = model(flipped_input)
            if isinstance(pred, tuple):
                pred = pred[0]
        pred = flip_fn(pred)  # unflip
        preds.append(pred)

    return torch.stack(preds, dim=0).mean(dim=0)

# ------------------ Postprocessing Functions -------------------

def apply_postprocessing(mask, method):
    if method == "none":
        return mask
    elif method == "closing":
        return binary_closing(mask, disk(3))
    elif method == "skeleton_repair":
        skel = skeletonize(mask)
        repaired = binary_closing(skel, disk(2))
        return repaired
    elif method == "remove_small":
        return remove_small_objects(mask.astype(bool), min_size=200)
    elif method == "closing+remove":
        closed = binary_closing(mask, disk(3))
        return remove_small_objects(closed.astype(bool), min_size=200)
    else:
        raise ValueError(f"Unknown postprocessing method: {method}")

# ------------------ Evaluation Wrapper ------------------------

def evaluate_with_postprocessing(model, test_loader, conf, method, output_dir, use_tta=False):
    logging.info(f"Evaluating with postprocessing: {method} {'+ TTA' if use_tta else ''}")
    out_dir = os.path.join(output_dir, method + ("_tta" if use_tta else ""))
    os.makedirs(out_dir, exist_ok=True)

    avg_loss, metrics = test_model(
        model, test_loader, conf, save_dir=out_dir, postprocess=method, use_tta=use_tta
    )
    return method, avg_loss, metrics

# ------------------ Modified test_model() with TTA ------------------------

def test_model(model, test_loader, conf, save_dir=None, return_metrics_only=False, postprocess="none", use_tta=False):
    model.eval()
    val_loss = 0.0
    patch_size = conf['dataset']['patch_size']
    batch_size = conf['test']['batch_size']

    use_topo = conf['loss'].get('use_topographic', False)
    loss_fn = get_loss_function(conf)
    all_metrics = {}

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image_patches = batch['image_patches'].cuda()
            mask_patches = batch['mask_patches'].cuda()
            coords = batch['coords']
            image_size = batch['image_size']
            image_idx = batch['image_idx']
            H_orig, W_orig = batch['original_size']
            B, N, C, H, W = image_patches.shape
            image_patches = image_patches.view(B * N, C, H, W)
            mask_patches = mask_patches.view(B * N, 1, H, W)

            if use_tta:
                preds = []
                for i in range(0, image_patches.size(0), batch_size):
                    patch_batch = image_patches[i:i + batch_size]
                    batch_preds = apply_tta(model, patch_batch)
                    preds.append(batch_preds)
                preds = torch.cat(preds, dim=0)
            else:
                preds = []
                for i in range(0, image_patches.size(0), batch_size):
                    patch_batch = image_patches[i:i + batch_size]
                    batch_preds = model(patch_batch)
                    if isinstance(batch_preds, tuple):
                        batch_preds = batch_preds[0]
                    preds.append(batch_preds)
                preds = torch.cat(preds, dim=0)

            # Loss
            if use_topo:
                distance_patches = batch['dist_patches'].cuda()
                distance_patches = distance_patches.view(B * N, 1, H, W)
                loss = loss_fn(preds, mask_patches, distance_patches)
            else:
                loss = loss_fn(preds, mask_patches)

            val_loss += loss.item()

            pred_full = reconstruct_from_patches(preds, coords, image_size, patch_size)
            mask_full = reconstruct_from_patches(mask_patches, coords, image_size, patch_size)

            pred_full = pred_full[:H_orig, :W_orig].cpu().numpy()
            mask_full = mask_full[:H_orig, :W_orig].cpu().numpy()

            # Binarize prediction
            pred_binary = (pred_full > 0.5).astype(np.uint8)

            # Apply postprocessing
            pred_processed = apply_postprocessing(pred_binary, postprocess)

            # Convert back to tensor for metric computation
            pred_tensor = torch.tensor(pred_processed).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor(mask_full).unsqueeze(0).unsqueeze(0)

            metrics = compute_metrics(pred_tensor, mask_tensor)
            metrics['clDice'] = compute_cldice(pred_tensor.squeeze(), mask_tensor.squeeze())

            for key in metrics:
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(metrics[key])

            if not return_metrics_only:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                img_path = os.path.join(test_loader.dataset.feature_dirs['enhanced'], batch['filename'][0])
                img = np.array(Image.open(img_path).convert("L"))

                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original Image')
                axes[1].imshow(mask_tensor.squeeze().numpy(), cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[2].imshow(pred_tensor.squeeze().numpy(), cmap='gray')
                axes[2].set_title("Prediction\n" + "\n".join([f"{k}: {metrics[k]:.3f}" for k in sorted(metrics)]))
                for ax in axes: ax.axis('off')
                plt.tight_layout()
                save_path = os.path.join(save_dir, "val_images", f'test_image_{idx}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()

            torch.cuda.empty_cache()

    avg_metrics = {k: np.nanmean(v) for k, v in all_metrics.items()}
    return val_loss / len(test_loader), avg_metrics

# ------------------ Main Entry -------------------------------

def main():
    args = parse_args()
    set_seed(42)

    # Load config
    conf = yaml.safe_load(open(args.config))

    if conf['loss'].get('use_topographic', False):
        conf['dataset']['dist_transform'] = True
    conf.setdefault('test', {})['use_tta'] = args.tta

    _, test_loader = get_patch_dataloaders(conf)

    model = torch.load(args.model, weights_only=False)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    os.makedirs(args.output, exist_ok=True)

    methods = ['none', 'closing', 'remove_small', 'closing+remove', 'skeleton_repair']

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Running test evaluation with TTA = {args.tta}")

    results = []
    for method in methods:
        method_name, loss, metrics = evaluate_with_postprocessing(
            model, test_loader, conf, method, args.output, use_tta=args.tta
        )
        results.append((method_name, loss, metrics))

    logging.info("Postprocessing Comparison Results:")
    for method, loss, metrics in results:
        logging.info(f"\n[{method.upper()}] Loss: {loss:.4f}")
        for k, v in metrics.items():
            logging.info(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
