import argparse, yaml, os, torch, numpy as np, random, logging, datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
from models import build_model
from datasets import get_patch_dataloaders
from metrics import compute_metrics
from utils import save_metadata, get_loss_function, reconstruct_from_patches
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clDice.cldice_metric.cldice import clDice as compute_cldice

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c','--config', required=True, help='Path to config YAML')
    p.add_argument('-e','--experiment_name', default='exp', help='Experiment name')
    return p.parse_args()


def test_model(model, test_loader, conf, save_dir=None, return_metrics_only=False):
    model.eval()
    dice_scores, iou_scores, cldice_scores = [], [], []
    val_loss = 0.0
    patch_size = conf['dataset']['patch_size']
    batch_size = conf['test']['batch_size']

    use_topo = conf['loss'].get('use_topographic', False)
    loss_fn = get_loss_function(conf)
    all_metrics = {}

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image_patches = batch['image_patches'].cuda()  # (N, C, H, W)
            mask_patches = batch['mask_patches'].cuda()    # (N, 1, H, W)
            coords = batch['coords']
            image_size = batch['image_size']
            image_idx = batch['image_idx']
            H_orig, W_orig = batch['original_size']
            B, N, C, H, W = image_patches.shape
            image_patches = image_patches.view(B * N, C, H, W)
            mask_patches = mask_patches.view(B * N, 1, H, W)

            preds = []
            for i in range(0, image_patches.size(0), batch_size):
                patch_batch = image_patches[i:i + batch_size]
                preds.append(model(patch_batch))
            preds = torch.cat(preds, dim=0)
            if use_topo:
                distance_patches = batch['dist_patches'].cuda()
                distance_patches = distance_patches.view(B * N, 1, H, W)
                loss = loss_fn(preds, mask_patches, distance_patches)
            else:
                loss = loss_fn(preds, mask_patches)

            val_loss += loss.item()

            pred_full = reconstruct_from_patches(preds, coords, image_size, patch_size)
            mask_full = reconstruct_from_patches(mask_patches, coords, image_size, patch_size)

            pred_full = pred_full[:H_orig, :W_orig].unsqueeze(0).unsqueeze(0)
            mask_full = mask_full[:H_orig, :W_orig].unsqueeze(0).unsqueeze(0)

            metrics = compute_metrics(pred_full, mask_full)
            metrics['clDice'] = compute_cldice(pred_full.squeeze(), mask_full.squeeze())

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
                axes[1].imshow(mask_full.squeeze().cpu().numpy(), cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[2].imshow(pred_full.squeeze().cpu().numpy(), cmap='gray')
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



def train_model(model, train_loader, test_loader, conf):
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6)
    best_dice, best_cldice, best_val_loss = 0.0, 0.0, float('inf')
    best_epoch, counter = 0, 0
    best_model_path = None
    accumulation_steps = conf['train'].get('accumulation_steps', 1)
    use_topo = conf['loss'].get('use_topographic', False)
    loss_fn = get_loss_function(conf)
    save_dir = conf['train']['save_dir']

    for epoch in range(1, conf['train']['num_epochs'] + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['inputs'].cuda()
            masks = batch['masks'].cuda()
            if use_topo:
                distances = batch['dist_maps'].cuda()
                loss = loss_fn(model(images), masks, distances) / accumulation_steps
            else:
                loss = loss_fn(model(images), masks) / accumulation_steps

            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, avg_metrics = test_model(model, test_loader, conf,  save_dir=save_dir, return_metrics_only=False)
        scheduler.step(avg_val_loss)

        if avg_metrics['Dice'] > best_dice:
            best_dice = avg_metrics['Dice']

        if avg_metrics['clDice'] > best_cldice:
            best_cldice = avg_metrics['clDice']

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            counter = 0
            #if best_model_path and os.path.exists(best_model_path):
            #    os.remove((best_model_path))
            os.makedirs(save_dir, exist_ok=True)
            save_name = f"best_model.pth"
            best_model_path = os.path.join(save_dir, save_name)
            #torch.save(model.state_dict(), os.path.join(save_dir, "best_model_state_dict.pth"))
            model.save_pretrained(model, os.path.join(save_dir, "best_model.pth"))
            
        else:
            counter += 1
            if counter >= conf['train']['patience']:
                logging.info(f"Early stopping at epoch {epoch} | Best Dice: {best_dice:.4f} at epoch {best_epoch}")
                return best_model_path, best_dice, best_cldice, best_val_loss, best_epoch

        if epoch % conf['train']['log_interval'] == 0 or epoch == conf['train']['num_epochs']:
            logging.info(
                f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | " +
                " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]) +
                f" | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )


    logging.info("Training completed.")
    return best_model_path, best_dice, best_cldice, best_val_loss, best_epoch

def main(conf):
    model = build_model(conf)
    train_loader, test_loader = get_patch_dataloaders(conf)
    best_model_path, best_dice, best_cldice, best_val_loss, best_epoch = train_model(model, train_loader, test_loader, conf)
    save_metadata(os.path.dirname(best_model_path), model, conf, best_dice, best_cldice, best_val_loss, best_epoch)
    #model.load_state_dict(torch.load(best_model_path))
    model = smp.from_pretrained(best_model_path, weights_only=False)
    save_dir = os.path.dirname(best_model_path)
    avg_val_loss, test_metrics = test_model(model, test_loader, conf, save_dir)
    logging.info(f"Test Loss: {avg_val_loss:.4f} | " + 
        " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))

if __name__ == '__main__':
    set_seed(42)
    args = parse_args()
    conf = yaml.safe_load(open(args.config))
    conf['train']['experiment_name'] = args.experiment_name
    conf['train']['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if conf['loss'].get('use_topographic', False):
        conf['dataset']['dist_transform'] = True
    save_dir = os.path.join(
        conf['train']['save_dir'], 
        conf['model']['name'],
        f"{conf['train']['timestamp']}_{conf['train']['experiment_name']}"
    )
    conf['train']['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"train_log.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info(f"Starting training with configuration: {conf}")
    logging.info(f"Experiment name: {args.experiment_name}")
    main(conf)
