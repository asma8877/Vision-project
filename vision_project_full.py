# -*- coding: utf-8 -*-
"""
EMCAD Vision Project — Camouflaged Object Detection on COD10K
Adapted from Google Colab for local/remote GPU execution.

Models:
  1. UNet (baseline)     — trained from scratch on COD10K train split
  2. EMCAD (PVTv2-B2)   — pretrained encoder, fine-tuned on COD10K train split

Data splits (from COD10K-v3 Train, camouflaged images only):
  Train : 80%  (~2432 images)
  Val   : 20%  (~608  images)  — used for early stopping only
  Test  : official COD10K-v3 Test set (2026 camouflaged images) — final report numbers

Usage:
    pip install -r requirements.txt
    python vision_project_full.py
"""

import os, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_DIR  = '/home/vteam1/COD10K-v3/Train'
TEST_DIR   = '/home/vteam1/COD10K-v3/Test'
OUTPUT_DIR = '/home/vteam1/outputs'

IMG_SIZE   = 320
BATCH_SIZE = 4
USE_EDGE   = True
NUM_EPOCHS = 50
LR         = 1e-4
PATIENCE   = 7
VAL_SPLIT  = 0.2   # 20% of train images used for validation
SEED       = 42
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class COD10KDataset(Dataset):
    """Loads camouflaged images only (skips empty GT masks)."""

    def __init__(self, dataset_dir, img_size=320,
                 use_edge=True, augment=False):
        self.img_size = img_size
        self.use_edge = use_edge
        self.augment  = augment

        self.img_dir  = os.path.join(dataset_dir, 'Image')
        self.mask_dir = os.path.join(dataset_dir, 'GT_Object')
        self.edge_dir = os.path.join(dataset_dir, 'GT_Edge')

        all_imgs = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            and ' (1).' not in f
        ])

        # Keep only images with a non-empty GT mask (camouflaged only)
        self.images = []
        for f in all_imgs:
            mask_path = os.path.join(self.mask_dir,
                                     os.path.splitext(f)[0] + '.png')
            if not os.path.exists(mask_path):
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.max() > 0:
                self.images.append(f)

        print(f"  {os.path.basename(dataset_dir)}: "
              f"{len(self.images)} camouflaged images")

    def __len__(self):
        return len(self.images)

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    def _load_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        return cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

    def _augment(self, img, mask, edge):
        if random.random() > 0.5:
            img  = cv2.flip(img,  1)
            mask = cv2.flip(mask, 1)
            edge = cv2.flip(edge, 1)
        if random.random() > 0.5:
            img  = cv2.flip(img,  0)
            mask = cv2.flip(mask, 0)
            edge = cv2.flip(edge, 0)
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D(
                (self.img_size // 2, self.img_size // 2), angle, 1.0)
            img  = cv2.warpAffine(img,  M, (self.img_size, self.img_size))
            mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size),
                                  flags=cv2.INTER_NEAREST)
            edge = cv2.warpAffine(edge, M, (self.img_size, self.img_size),
                                  flags=cv2.INTER_NEAREST)
        if random.random() > 0.5:
            alpha = random.uniform(0.7, 1.3)
            beta  = random.randint(-20, 20)
            img   = np.clip(img.astype(np.float32) * alpha + beta,
                            0, 255).astype(np.uint8)
        if random.random() > 0.5:
            scale = random.uniform(0.75, 1.0)
            h = w = int(self.img_size * scale)
            x = random.randint(0, self.img_size - w)
            y = random.randint(0, self.img_size - h)
            img  = cv2.resize(img[y:y+h, x:x+w],
                              (self.img_size, self.img_size))
            mask = cv2.resize(mask[y:y+h, x:x+w],
                              (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)
            edge = cv2.resize(edge[y:y+h, x:x+w],
                              (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)
        if random.random() > 0.5:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img   = np.clip(img.astype(np.float32) + noise,
                            0, 255).astype(np.uint8)
        return img, mask, edge

    def _to_tensor(self, img, mask, edge):
        img  = img.astype(np.float32) / 255.0
        img  = (img - np.array([0.485, 0.456, 0.406])) / \
                     np.array([0.229, 0.224, 0.225])
        img_t  = torch.from_numpy(img).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(
            (mask > 127).astype(np.float32)).unsqueeze(0)
        edge_t = torch.from_numpy(
            (edge > 127).astype(np.float32)).unsqueeze(0)
        return img_t, mask_t, edge_t

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        stem      = os.path.splitext(img_name)[0]
        mask_name = stem + '.png'

        img  = self._load_image(os.path.join(self.img_dir,  img_name))
        mask = self._load_mask(os.path.join(self.mask_dir, mask_name))
        edge_path = os.path.join(self.edge_dir, mask_name)
        edge = self._load_mask(edge_path) if os.path.exists(edge_path) \
               else np.zeros_like(mask)

        if self.augment:
            img, mask, edge = self._augment(img, mask, edge)

        img_t, mask_t, edge_t = self._to_tensor(img, mask, edge)

        if self.use_edge:
            return img_t, mask_t, edge_t, img_name
        return img_t, mask_t, img_name


# ---------------------------------------------------------------------------
# Subset wrappers (train with augment, val without)
# ---------------------------------------------------------------------------
class AugmentedSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        self.dataset.augment = True
        item = self.dataset[self.indices[idx]]
        self.dataset.augment = False
        return item

class CleanSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        self.dataset.augment = False
        return self.dataset[self.indices[idx]]


# ---------------------------------------------------------------------------
# Data loading with 80/20 train/val split
# ---------------------------------------------------------------------------
print("\nLoading datasets...")

full_train_ds = COD10KDataset(TRAIN_DIR, img_size=IMG_SIZE,
                              use_edge=USE_EDGE, augment=False)
test_ds       = COD10KDataset(TEST_DIR,  img_size=IMG_SIZE,
                              use_edge=USE_EDGE, augment=False)

n_total = len(full_train_ds)
indices = np.random.permutation(n_total).tolist()
n_val   = int(n_total * VAL_SPLIT)
val_idx   = indices[:n_val]
train_idx = indices[n_val:]

print(f"  Split → Train: {len(train_idx)} | Val: {len(val_idx)} | "
      f"Test (official): {len(test_ds)}")

train_subset = AugmentedSubset(full_train_ds, train_idx)
val_subset   = CleanSubset(full_train_ds,   val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,      batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"  Batches → Train: {len(train_loader)} | "
      f"Val: {len(val_loader)} | Test: {len(test_loader)}")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

def denorm(t):
    img = t.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * std + mean, 0, 1)

# Save sample grid
num_samples = 5
sample_indices = random.sample(range(len(full_train_ds)), num_samples)
fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
fig.suptitle('Training Samples', fontsize=13, fontweight='bold')
full_train_ds.augment = False
for row, idx in enumerate(sample_indices):
    img, mask, edge, name = full_train_ds[idx]
    img_np  = denorm(img)
    mask_np = mask.squeeze().numpy()
    edge_np = edge.squeeze().numpy()
    overlay = img_np.copy()
    overlay[mask_np > 0.5] = [1.0, 0.2, 0.2]
    for col, (data, title) in enumerate(zip(
            [img_np, mask_np, edge_np, overlay],
            ['Image', 'GT Mask', 'Edge GT', 'Overlay'])):
        axes[row, col].imshow(data, cmap='gray' if col in [1, 2] else None)
        axes[row, col].set_title(
            f"{title}\n{name if col==0 else ''}", fontsize=8)
        axes[row, col].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'train_samples.png'), dpi=100)
plt.close()
print("Training sample grid saved.")


# ---------------------------------------------------------------------------
# Shared loss and metric functions
# ---------------------------------------------------------------------------
def bce_dice_loss(pred, target, eps=1e-6):
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    prob = torch.sigmoid(pred)
    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice  = 1 - (2 * inter + eps) / (union + eps)
    return bce + dice.mean()


def compute_metrics(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    p  = preds.view(-1);  t  = targets.view(-1);  pr = probs.view(-1)
    tp = (p * t).sum();   fp = (p * (1 - t)).sum();  fn = ((1 - p) * t).sum()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    f_measure = (1.3 * precision * recall) / (0.3 * precision + recall + eps)
    fg_mean   = pr[t == 1].mean() if (t == 1).any() else torch.tensor(0.0)
    bg_mean   = 1 - pr[t == 0].mean() if (t == 0).any() else torch.tensor(0.0)
    s_measure = 0.5 * fg_mean + 0.5 * bg_mean
    mu_p = pr.mean(); mu_t = t.mean()
    align = 2 * (pr - mu_p) * (t - mu_t) / \
            (((pr - mu_p)**2 + (t - mu_t)**2).clamp(min=eps))
    e_measure = ((1 + align) ** 2 / 4).mean()
    mae       = (probs - targets).abs().mean()
    return {
        'dice':      dice.item(),
        'iou':       iou.item(),
        'precision': precision.item(),
        'recall':    recall.item(),
        'f_measure': f_measure.item(),
        's_measure': s_measure.item(),
        'e_measure': e_measure.item(),
        'mae':       mae.item(),
    }


def evaluate(model, loader, device, get_logits_fn):
    model.eval()
    totals = {k: 0.0 for k in
              ['dice','iou','precision','recall',
               'f_measure','s_measure','e_measure','mae']}
    n = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            imgs, masks, edges, _ = batch
            imgs  = imgs.to(device)
            masks = masks.to(device)
            out    = model(imgs)
            logits = get_logits_fn(out)
            m = compute_metrics(logits, masks)
            for k in totals:
                totals[k] += m[k]
            n += 1
    return {k: v / n for k, v in totals.items()}


def save_prediction_grid(model, loader, device, title, save_name,
                         get_logits_fn, num_samples=5):
    model.eval()
    all_items = []
    with torch.no_grad():
        for batch in loader:
            imgs, masks, edges, names = batch
            imgs = imgs.to(device)
            out  = model(imgs)
            logits = get_logits_fn(out)
            p1 = torch.sigmoid(logits).cpu()
            for i in range(len(names)):
                all_items.append(
                    (imgs[i].cpu(), masks[i], p1[i], names[i]))
            if len(all_items) >= num_samples * 3:
                break
    samples = random.sample(all_items, min(num_samples, len(all_items)))
    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 4 * len(samples)))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for row, (img, mask, pred, name) in enumerate(samples):
        img_np   = np.clip(img.permute(1,2,0).numpy() * std + mean, 0, 1)
        mask_np  = mask.squeeze().numpy()
        pred_np  = pred.squeeze().numpy()
        pred_bin = (pred_np > 0.5).astype(np.float32)
        overlay  = img_np.copy()
        overlay[pred_bin > 0.5] = [1.0, 0.2, 0.2]
        for col, (data, ttl) in enumerate(zip(
                [img_np, mask_np, pred_np, overlay],
                ['Input', 'Ground Truth', 'Prediction', 'Overlay'])):
            axes[row, col].imshow(
                data, cmap='gray' if col in [1, 2] else None)
            axes[row, col].set_title(
                f"{ttl}\n{name if col==0 else ''}", fontsize=8)
            axes[row, col].axis('off')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Plot saved to: {path}")


def error_analysis(model, loader, device, get_logits_fn,
                   num_samples=6, save_name="error_analysis.png"):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loader:
            imgs, masks, edges, names = batch
            imgs  = imgs.to(device)
            masks = masks.to(device)
            out   = model(imgs)
            logits = get_logits_fn(out)
            p1 = torch.sigmoid(logits)
            for i in range(len(names)):
                m = compute_metrics(p1[i].unsqueeze(0), masks[i].unsqueeze(0))
                results.append({
                    'img':  imgs[i].cpu(), 'mask': masks[i].cpu(),
                    'pred': p1[i].cpu(),   'name': names[i],
                    'dice': m['dice']
                })
    results.sort(key=lambda x: x['dice'])
    half   = num_samples // 2
    show   = results[:half] + results[-half:]
    labels = ([f'Fail Dice={r["dice"]:.3f}' for r in results[:half]] +
              [f'Pass Dice={r["dice"]:.3f}' for r in results[-half:]])
    fig, axes = plt.subplots(len(show), 4, figsize=(16, 4 * len(show)))
    fig.suptitle('Error Analysis: Worst (top) vs Best (bottom)',
                 fontsize=13, fontweight='bold')
    for row, (r, lbl) in enumerate(zip(show, labels)):
        img_np   = np.clip(
            r['img'].permute(1,2,0).numpy() * std + mean, 0, 1)
        mask_np  = r['mask'].squeeze().numpy()
        pred_np  = r['pred'].squeeze().numpy()
        pred_bin = (pred_np > 0.5).astype(np.float32)
        overlay  = img_np.copy()
        overlay[pred_bin > 0.5] = [1.0, 0.2, 0.2]
        for col, (data, ttl) in enumerate(zip(
                [img_np, mask_np, pred_np, overlay],
                ['Input', 'Ground Truth', 'Prediction', 'Overlay'])):
            axes[row, col].imshow(
                data, cmap='gray' if col in [1, 2] else None)
            axes[row, col].set_title(
                f"{ttl}\n{lbl if col==0 else ''}", fontsize=7)
            axes[row, col].axis('off')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"Error analysis saved to: {path}")


def save_training_curves(train_losses, val_losses, val_dices, zeroshot_dice,
                         title, save_name):
    epochs_ran = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Left: train loss vs val loss on same plot
    ax1.plot(epochs_ran, train_losses, 'b-o', markersize=3, label='Train Loss')
    ax1.plot(epochs_ran, val_losses,   'r-o', markersize=3, label='Val Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} Train vs Val Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    # Right: val dice with zero-shot reference line
    ax2.plot(epochs_ran, val_dices, 'g-o', markersize=3, label='Val Dice')
    if zeroshot_dice > 0:
        ax2.axhline(y=zeroshot_dice, color='r', linestyle='--',
                    label=f"Zero-Shot ({zeroshot_dice:.4f})")
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice Score')
    ax2.set_title(f'{title} Validation Dice')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle(f'{title} Fine-Tuning on COD10K',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=100)
    plt.close()
    print(f"Training curves saved: {save_name}")


def train_model(model, loss_fn, get_logits_fn, model_name):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    train_losses = []; val_losses = []; val_dices = []
    best_dice = 0.0; patience_cnt = 0; best_weights = None

    print(f"\nFine-tuning {model_name} | Epochs: {NUM_EPOCHS} | "
          f"Patience: {PATIENCE}")
    print("=" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader,
                          desc=f"{model_name} Epoch {epoch}/{NUM_EPOCHS}",
                          leave=False):
            imgs, masks, edges, _ = batch
            imgs  = imgs.to(device)
            masks = masks.to(device)
            edges = edges.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = loss_fn(out, masks, edges)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Compute val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs, masks, edges, _ = batch
                imgs  = imgs.to(device)
                masks = masks.to(device)
                edges = edges.to(device)
                out  = model(imgs)
                val_loss += loss_fn(out, masks, edges).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_m = evaluate(model, val_loader, device, get_logits_fn)
        val_dices.append(val_m['dice'])

        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Dice: {val_m['dice']:.4f} | "
              f"Val IoU: {val_m['iou']:.4f}")

        if val_m['dice'] > best_dice:
            best_dice    = val_m['dice']
            best_weights = {k: v.clone()
                            for k, v in model.state_dict().items()}
            patience_cnt = 0
            print(f"  --> Best Val Dice: {best_dice:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    print(f"\n{model_name} training complete. Best Val Dice: {best_dice:.4f}")
    return train_losses, val_losses, val_dices, best_dice


# ===========================================================================
# MODEL 1: UNet Baseline
# ===========================================================================
print("\n" + "=" * 60)
print("  MODEL 1: UNet Baseline")
print("=" * 60)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                       kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2,
                         diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class UNetWithEdge(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.inc   = DoubleConv(in_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1   = Up(1024, 512)
        self.up2   = Up(512,  256)
        self.up3   = Up(256,  128)
        self.up4   = Up(128,  64)
        self.mask_head = nn.Conv2d(64, 1, 1)
        self.edge_head = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        x1 = self.inc(x);   x2 = self.down1(x1)
        x3 = self.down2(x2); x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4); x = self.up2(x, x3)
        x  = self.up3(x,  x2); x = self.up4(x, x1)
        return self.mask_head(x), self.edge_head(x)

def unet_loss(out, mask, edge):
    mask_logits, edge_logits = out
    return (bce_dice_loss(mask_logits, mask) +
            0.5 * F.binary_cross_entropy_with_logits(edge_logits, edge))

unet_get_logits = lambda out: out[0]

unet = UNetWithEdge().to(device)
print(f"UNet parameters: {sum(p.numel() for p in unet.parameters())/1e6:.2f}M")

unet_train_losses, unet_val_losses, unet_val_dices, unet_best_val_dice = train_model(
    unet, unet_loss, unet_get_logits, "UNet")

print("\nUNet final test evaluation...")
unet_test_results = evaluate(unet, test_loader, device, unet_get_logits)
print("UNet Fine-Tuned Test Results:")
for k, v in unet_test_results.items():
    print(f"  {k:<12}: {v:.4f}")

save_prediction_grid(unet, test_loader, device,
                     title="UNet Fine-Tuned Predictions",
                     save_name="unet_predictions.png",
                     get_logits_fn=unet_get_logits)
save_training_curves(unet_train_losses, unet_val_losses, unet_val_dices,
                     0.0, "UNet", "unet_training_curves.png")
error_analysis(unet, test_loader, device, unet_get_logits,
               save_name="unet_error_analysis.png")

torch.save({
    'model_state_dict': unet.state_dict(),
    'test_metrics':     unet_test_results,
    'best_val_dice':    unet_best_val_dice,
}, os.path.join(OUTPUT_DIR, 'unet_cod10k.pth'))
print("UNet checkpoint saved.")


# ===========================================================================
# MODEL 2: EMCAD (PVTv2-B2 encoder)
# ===========================================================================
print("\n" + "=" * 60)
print("  MODEL 2: EMCAD (PVTv2-B2)")
print("=" * 60)

import timm

class MSCAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1,
                             groups=channels, bias=False)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=2,
                             groups=channels, bias=False)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=3,
                             groups=channels, bias=False)
        self.pw  = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.bn  = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca_avg = nn.AdaptiveAvgPool2d(1)
        self.ca_fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // 4), nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels), nn.Sigmoid()
        )
    def forward(self, x):
        ms = self.relu(self.bn(self.pw(
            torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1))))
        ca = self.ca_fc(self.ca_avg(ms)).view(ms.size(0), -1, 1, 1)
        return ms * ca + x

class EUCB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.dw   = nn.Conv2d(out_ch, out_ch, 3, padding=1,
                              groups=out_ch, bias=False)
        self.pw   = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.pw(self.dw(self.up(x)))))

class LGAG(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1,
                      groups=max(1, channels // 8), bias=False),
            nn.BatchNorm2d(channels), nn.Sigmoid()
        )
    def forward(self, x, skip):
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        g = self.gate(torch.cat([x, skip], dim=1))
        return x * g + skip * (1 - g)

class EMCADDecoder(nn.Module):
    def __init__(self, enc_channels, dec_ch=128):
        super().__init__()
        c1, c2, c3, c4 = enc_channels
        self.proj3 = nn.Conv2d(c3, dec_ch, 1, bias=False)
        self.proj2 = nn.Conv2d(c2, dec_ch, 1, bias=False)
        self.proj1 = nn.Conv2d(c1, dec_ch, 1, bias=False)
        self.eucb4 = EUCB(c4, dec_ch);   self.eucb3 = EUCB(dec_ch, dec_ch)
        self.eucb2 = EUCB(dec_ch, dec_ch); self.eucb1 = EUCB(dec_ch, dec_ch)
        self.lgag3 = LGAG(dec_ch); self.lgag2 = LGAG(dec_ch)
        self.lgag1 = LGAG(dec_ch)
        self.mscam4 = MSCAM(dec_ch); self.mscam3 = MSCAM(dec_ch)
        self.mscam2 = MSCAM(dec_ch); self.mscam1 = MSCAM(dec_ch)
        self.head4 = nn.Conv2d(dec_ch, 1, 1)
        self.head3 = nn.Conv2d(dec_ch, 1, 1)
        self.head2 = nn.Conv2d(dec_ch, 1, 1)
        self.head1 = nn.Conv2d(dec_ch, 1, 1)
        self.edge_head = nn.Conv2d(dec_ch, 1, 1)
    def forward(self, feats, img_size):
        f1, f2, f3, f4 = feats
        d4 = self.mscam4(self.eucb4(f4))
        d3 = self.mscam3(self.lgag3(d4, self.proj3(f3))); d3 = self.eucb3(d3)
        d2 = self.mscam2(self.lgag2(d3, self.proj2(f2))); d2 = self.eucb2(d2)
        d1 = self.mscam1(self.lgag1(d2, self.proj1(f1))); d1 = self.eucb1(d1)
        up = lambda x: F.interpolate(x, size=img_size,
                                     mode='bilinear', align_corners=False)
        return (up(self.head1(d1)), up(self.head2(d2)),
                up(self.head3(d3)), up(self.head4(d4)),
                up(self.edge_head(d1)))

class EMCADModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            'pvt_v2_b2', pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3)
        )
        enc_ch = self.encoder.feature_info.channels()
        print(f"Encoder channels: {enc_ch}")
        self.decoder = EMCADDecoder(enc_channels=enc_ch, dec_ch=128)
    def forward(self, x):
        img_size = (x.shape[2], x.shape[3])
        return self.decoder(self.encoder(x), img_size)

def emcad_loss(preds, mask, edge):
    p1, p2, p3, p4, pe = preds
    loss = (bce_dice_loss(p1, mask) +
            0.8 * bce_dice_loss(p2, mask) +
            0.6 * bce_dice_loss(p3, mask) +
            0.4 * bce_dice_loss(p4, mask))
    return loss + 0.5 * F.binary_cross_entropy_with_logits(pe, edge)

emcad_get_logits = lambda out: out[0]

print("Building EMCAD model with PVTv2-B2 encoder...")
emcad = EMCADModel(pretrained=True).to(device)
print(f"EMCAD parameters: "
      f"{sum(p.numel() for p in emcad.parameters())/1e6:.2f}M")

print("\nRunning EMCAD zero-shot baseline...")
emcad_zeroshot = evaluate(emcad, test_loader, device, emcad_get_logits)
print("EMCAD Zero-Shot Results:")
for k, v in emcad_zeroshot.items():
    print(f"  {k:<12}: {v:.4f}")

save_prediction_grid(emcad, test_loader, device,
                     title="EMCAD Zero-Shot Predictions",
                     save_name="emcad_zeroshot_predictions.png",
                     get_logits_fn=emcad_get_logits)

emcad_train_losses, emcad_val_losses, emcad_val_dices, emcad_best_val_dice = train_model(
    emcad, emcad_loss, emcad_get_logits, "EMCAD")

print("\nEMCAD final test evaluation...")
emcad_test_results = evaluate(emcad, test_loader, device, emcad_get_logits)
print("\nEMCAD Fine-Tuned Test Results:")
print("-" * 45)
for k, v in emcad_test_results.items():
    print(f"  {k:<12}: {v:.4f}")

save_prediction_grid(emcad, test_loader, device,
                     title="EMCAD Fine-Tuned Predictions",
                     save_name="emcad_finetuned_predictions.png",
                     get_logits_fn=emcad_get_logits)
save_training_curves(emcad_train_losses, emcad_val_losses, emcad_val_dices,
                     emcad_zeroshot['dice'], "EMCAD", "emcad_training_curves.png")
error_analysis(emcad, test_loader, device, emcad_get_logits,
               save_name="emcad_error_analysis.png")

torch.save({
    'model_state_dict':  emcad.state_dict(),
    'zeroshot_metrics':  emcad_zeroshot,
    'finetuned_metrics': emcad_test_results,
    'best_val_dice':     emcad_best_val_dice,
}, os.path.join(OUTPUT_DIR, 'emcad_finetuned_cod10k.pth'))
print("EMCAD checkpoint saved.")


# ===========================================================================
# Comparison Table + Bar Chart
# ===========================================================================
print("\n" + "=" * 65)
print("       Comparison Table on COD10K Test Set")
print("=" * 65)

published = {
    'SINet [Fan 2020]':    {'dice':0.712,'iou':0.599,'f_measure':0.706,
                            's_measure':0.771,'e_measure':0.806,'mae':0.051},
    'PFNet [Mei 2021]':    {'dice':0.793,'iou':0.695,'f_measure':0.791,
                            's_measure':0.800,'e_measure':0.868,'mae':0.040},
    'SegMaR [Jia 2022]':   {'dice':0.815,'iou':0.719,'f_measure':0.815,
                            's_measure':0.815,'e_measure':0.875,'mae':0.035},
    'ZoomNet [Pang 2022]': {'dice':0.820,'iou':0.729,'f_measure':0.820,
                            's_measure':0.820,'e_measure':0.892,'mae':0.029},
}

keys = ['dice', 'iou', 'f_measure', 's_measure', 'e_measure', 'mae']
rows = {name: {k: v[k] for k in keys} for name, v in published.items()}
rows['UNet Fine-Tuned (Ours)']  = {k: unet_test_results[k]  for k in keys}
rows['EMCAD Zero-Shot (Ours)']  = {k: emcad_zeroshot[k]     for k in keys}
rows['EMCAD Fine-Tuned (Ours)'] = {k: emcad_test_results[k] for k in keys}

df = pd.DataFrame(rows).T.round(4)
df.columns = ['Dice', 'IoU', 'F-measure', 'S-measure', 'E-measure', 'MAE']
print(df.to_string())
print("=" * 65)

metrics_to_plot = ['Dice', 'IoU', 'F-measure', 'S-measure', 'E-measure']
methods = list(df.index)
x      = np.arange(len(metrics_to_plot))
width  = 0.10
colors = ['#4472C4','#ED7D31','#A9D18E','#FFC000',
          '#FF0000','#B19CD9','#7030A0']

fig, ax = plt.subplots(figsize=(17, 6))
for i, (method, color) in enumerate(zip(methods, colors)):
    vals = [df.loc[method, m] for m in metrics_to_plot]
    ax.bar(x + i * width, vals, width, label=method,
           color=color, alpha=0.85)
ax.set_xticks(x + width * (len(methods) - 1) / 2)
ax.set_xticklabels(metrics_to_plot, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('UNet & EMCAD vs Published Baselines on COD10K',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_bar_chart.png'), dpi=100)
plt.close()
print("Comparison bar chart saved.")


# ===========================================================================
# Side-by-side comparison grid: UNet vs EMCAD Zero-Shot vs EMCAD Fine-Tuned
# ===========================================================================
print("Generating side-by-side comparison grid...")

def get_predictions(model, loader, device, get_logits_fn, n=5):
    """Collect n (image, mask, pred) tuples from loader."""
    model.eval()
    items = []
    with torch.no_grad():
        for batch in loader:
            imgs, masks, edges, names = batch
            imgs = imgs.to(device)
            out  = model(imgs)
            logits = get_logits_fn(out)
            preds  = torch.sigmoid(logits).cpu()
            for i in range(len(names)):
                items.append((imgs[i].cpu(), masks[i], preds[i], names[i]))
            if len(items) >= n:
                break
    return items[:n]

# Use same fixed indices so all three models see identical images
fixed_loader = DataLoader(
    CleanSubset(test_ds, list(range(min(20, len(test_ds))))),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

n_compare = 5
unet_preds   = get_predictions(unet,  fixed_loader, device, unet_get_logits,  n_compare)
zshot_preds  = get_predictions(emcad, fixed_loader, device, emcad_get_logits, n_compare)

# Reload best EMCAD weights (already loaded) — fine-tuned preds
ftune_preds  = get_predictions(emcad, fixed_loader, device, emcad_get_logits, n_compare)

# columns: Input | GT | UNet | EMCAD ZS | EMCAD FT
col_titles = ['Input', 'Ground Truth', 'UNet', 'EMCAD Zero-Shot', 'EMCAD Fine-Tuned']
fig, axes  = plt.subplots(n_compare, 5, figsize=(20, 4 * n_compare))
fig.suptitle('Side-by-Side Comparison: UNet vs EMCAD Zero-Shot vs EMCAD Fine-Tuned',
             fontsize=13, fontweight='bold')

for row in range(n_compare):
    img_np  = np.clip(
        unet_preds[row][0].permute(1,2,0).numpy() * std + mean, 0, 1)
    mask_np = unet_preds[row][1].squeeze().numpy()
    unet_np = unet_preds[row][2].squeeze().numpy()
    zs_np   = zshot_preds[row][2].squeeze().numpy()
    ft_np   = ftune_preds[row][2].squeeze().numpy()

    for col, (data, title) in enumerate(zip(
            [img_np, mask_np, unet_np, zs_np, ft_np], col_titles)):
        axes[row, col].imshow(
            data, cmap='gray' if col in [1, 2, 3, 4] else None)
        if row == 0:
            axes[row, col].set_title(title, fontsize=10, fontweight='bold')
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'side_by_side_comparison.png'), dpi=100)
plt.close()
print("Side-by-side comparison grid saved.")

print("\nAll done! Outputs saved to:", OUTPUT_DIR)