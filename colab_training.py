# ============================================================================
# 🧠 RTH U-Net Training — Google Colab Version
# ============================================================================
# Monitoring Revolusi Ruang Terbuka Hijau
# Arsitektur U-Net pada Citra Satelit Sentinel-2
#
# INSTRUCTIONS:
# 1. Upload your dataset folder to Google Drive
# 2. Copy each section (separated by # %% comments) into separate Colab cells
# 3. Run cells sequentially
# 4. Download the trained model .pth file or find it in your Drive
#
# REQUIRED DATA ON GOOGLE DRIVE:
# ──────────────────────────────
# Your Google Drive should have this structure:
#
#   MyDrive/
#   └── Monitoring_RTH/                        ← BASE_DIR (configurable below)
#       └── data/
#           └── Dataset_UNet_v2_improved/
#               ├── images/                    ← .npy patch files (5-band: R,G,B,NIR,NDVI)
#               └── masks/                     ← .npy mask files (binary RTH masks)
#
# HOW TO PREPARE:
# ──────────────────────────────
# Option A: Upload your existing local dataset
#   - Zip the 'Dataset_UNet_v2_improved' folder from your local project
#   - Upload zip to Google Drive, then unzip in Colab
#
# Option B: Run patch creation on Colab (if you have the GeoTIFFs)
#   - Upload sentinel2_jakarta_2019.tif, sentinel2_jakarta_2023.tif
#   - Upload the shapefile folder (jakarta_rth_filtered.shp + .shx, .dbf, .prj)
#   - Run prepare_patches_improved.py first
#
# OUTPUT:
# ──────────────────────────────
# The trained model will be saved to:
#   MyDrive/Monitoring_RTH/models/active_model.pth
#
# You can download this .pth file and upload it to your Streamlit dashboard
# using the "📂 Unggah Model Pre-Trained" tab.
# ============================================================================


# %% [1] ── Install Dependencies ─────────────────────────────────────────────
# Run this cell first to install required packages

!pip install -q albumentations scikit-learn tqdm


# %% [2] ── Mount Google Drive ───────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')


# %% [3] ── Configuration ────────────────────────────────────────────────────
# ⚠️ EDIT THIS SECTION to match your Google Drive folder structure

import os

# ┌─────────────────────────────────────────────────────────────┐
# │  EDIT THESE PATHS TO MATCH YOUR GOOGLE DRIVE STRUCTURE      │
# └─────────────────────────────────────────────────────────────┘
#
# 👉 DATA YOU NEED TO UPLOAD TO GOOGLE DRIVE:
# 1. Create a folder named "Monitoring_RTH" in your Drive.
# 2. Inside it, create "data", then "Dataset_UNet_v2_improved".
# 3. Upload ALL your `.npy` patch files into:
#      MyDrive/Monitoring_RTH/data/Dataset_UNet_v2_improved/images/
# 4. Upload ALL your `.npy` mask files into:
#      MyDrive/Monitoring_RTH/data/Dataset_UNet_v2_improved/masks/
#
# Easiest way: Just ZIP your local `data/Dataset_UNet_v2_improved`
# folder, upload the zip to Drive, and extract it there.

# Base directory in Google Drive (where your project data lives)
BASE_DIR = "/content/drive/MyDrive/Monitoring_RTH"

# Path to the dataset folder containing images/ and masks/ subdirectories
DATASET_DIR = os.path.join(BASE_DIR, "data", "Dataset_UNet_v2_improved")

# Directory where the trained model will be saved
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")

# ┌─────────────────────────────────────────────────────────────┐
# │  HYPERPARAMETERS — defaults match the research notebook     │
# └─────────────────────────────────────────────────────────────┘

CONFIG = {
    'patch_size': 256,
    'in_channels': 5,         # R, G, B, NIR, NDVI
    'out_channels': 1,        # Binary segmentation (RTH / non-RTH)
    'batch_size': 8,          # Increase to 16 or 32 if GPU memory allows
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience_lr': 5,         # ReduceLROnPlateau patience
    'patience_es': 10,        # Early stopping patience
    'seed': 42,
    'num_workers': 2,         # DataLoader workers (2 is good for Colab)
}

print("✅ Configuration loaded")
print(f"   Dataset: {DATASET_DIR}")
print(f"   Model output: {MODEL_OUTPUT_DIR}")


# %% [4] ── Imports & Setup ──────────────────────────────────────────────────

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ── Set seed for reproducibility ─────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])

# ── Device ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("   ⚠️  No GPU detected! Training will be very slow.")
    print("   Go to Runtime → Change runtime type → GPU (T4)")


# %% [5] ── U-Net Architecture ───────────────────────────────────────────────
# Exact same architecture as the Streamlit dashboard (utils.py)

class DoubleConv(nn.Module):
    """Double convolution block: Conv2d → BN → ReLU → Conv2d → BN → ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for RTH segmentation.
    Architecture: 4 encoder blocks → bottleneck → 4 decoder blocks
    Channels: 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64
    """
    def __init__(self, in_ch=5, out_ch=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        self.drop_bn = nn.Dropout2d(0.5)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.drop4 = nn.Dropout2d(0.3)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.drop3 = nn.Dropout2d(0.3)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.drop2 = nn.Dropout2d(0.3)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.drop_bn(self.bottleneck(self.pool(e4)))
        d4 = self.drop4(self.dec4(torch.cat([self.up4(b), e4], 1)))
        d3 = self.drop3(self.dec3(torch.cat([self.up3(d4), e3], 1)))
        d2 = self.drop2(self.dec2(torch.cat([self.up2(d3), e2], 1)))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out_conv(d1))

# Quick test
model_test = UNet(in_ch=5, out_ch=1)
total_params = sum(p.numel() for p in model_test.parameters())
print(f"✅ UNet loaded — {total_params:,} parameters ({total_params/1e6:.1f}M)")
del model_test


# %% [6] ── Loss Functions ───────────────────────────────────────────────────
# BCE + Dice Loss — same as notebook and dashboard

def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


def bce_dice_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    return bce + dice_loss(pred, target)


# %% [7] ── RTH Dataset ──────────────────────────────────────────────────────
# Loads .npy patches — exact same normalization as dashboard RTHDataset

class RTHDataset(Dataset):
    """
    Loads .npy patch files from Dataset_UNet_v2_improved/images and masks.

    Normalization (matches dashboard exactly):
      Band 0-3 (R,G,B,NIR): ÷ 10000
      Band 4 (NDVI): clamp(-1,1) → (v+1)/2
    """
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img  = np.load(self.image_paths[idx]).astype(np.float32)
        mask = np.load(self.mask_paths[idx]).astype(np.float32)

        # Normalize bands (same as utils.py)
        img[:4]  = img[:4] / 10000.0
        img[4]   = (np.clip(img[4], -1.0, 1.0) + 1.0) / 2.0

        # Convert to HWC for albumentations
        img_hwc = np.transpose(img, (1, 2, 0))

        if self.transform:
            aug     = self.transform(image=img_hwc, mask=mask)
            img_hwc = aug['image']
            mask    = aug['mask']

        # Convert back to CHW tensor
        img_t  = torch.from_numpy(np.transpose(img_hwc, (2, 0, 1)).copy())
        mask_t = torch.from_numpy(mask.copy()).unsqueeze(0)
        return img_t, mask_t


# %% [8] ── Load Dataset & Create DataLoaders ────────────────────────────────

# ── Verify dataset exists ─────────────────────────────
image_dir = os.path.join(DATASET_DIR, "images")
mask_dir  = os.path.join(DATASET_DIR, "masks")

assert os.path.isdir(image_dir), f"❌ Image directory not found: {image_dir}"
assert os.path.isdir(mask_dir),  f"❌ Mask directory not found: {mask_dir}"

imgs  = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.npy')])
masks = sorted([os.path.join(mask_dir,  f) for f in os.listdir(mask_dir)  if f.endswith('.npy')])

assert len(imgs) == len(masks), f"❌ Mismatch: {len(imgs)} images vs {len(masks)} masks"
print(f"📦 Dataset found: {len(imgs):,} patches")

# ── Train / Val split (70/15/15 — same as notebook) ──
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
    imgs, masks, test_size=0.30, random_state=CONFIG['seed']
)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(
    temp_imgs, temp_masks, test_size=0.50, random_state=CONFIG['seed']
)

print(f"   Train: {len(train_imgs):,} | Val: {len(val_imgs):,} | Test: {len(test_imgs):,}")

# ── Augmentation pipeline ────────────────────────────
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.GaussNoise(p=0.2),
])

# ── Create datasets & loaders ────────────────────────
train_ds = RTHDataset(train_imgs, train_masks, transform=train_aug)
val_ds   = RTHDataset(val_imgs, val_masks)
test_ds  = RTHDataset(test_imgs, test_masks)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                          num_workers=CONFIG['num_workers'], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG['batch_size'], shuffle=False,
                          num_workers=CONFIG['num_workers'], pin_memory=True)

print(f"   Train batches/epoch: {len(train_loader)}")
print(f"   Val batches/epoch:   {len(val_loader)}")
print(f"   Test batches:        {len(test_loader)}")

# ── Quick sanity check ───────────────────────────────
x_sample, y_sample = next(iter(train_loader))
print(f"\n🔍 Sample batch shape: image={x_sample.shape}, mask={y_sample.shape}")
print(f"   Image value range: [{x_sample.min():.3f}, {x_sample.max():.3f}]")
print(f"   Mask unique values: {torch.unique(y_sample).tolist()}")


# %% [9] ── Training Loop ────────────────────────────────────────────────────

def compute_iou(pred, target, thresh=0.5):
    """Compute IoU (Intersection over Union) metric."""
    pred_bin = (pred > thresh).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return (inter / (union + 1e-6)).item()


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch, return (avg_loss, avg_iou)."""
    model.train()
    total_loss, total_iou = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = bce_dice_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_iou  += compute_iou(out.detach(), y)

    n = len(loader)
    return total_loss / n, total_iou / n


@torch.no_grad()
def validate(model, loader, device):
    """Validate model, return (avg_loss, avg_iou)."""
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += bce_dice_loss(out, y).item()
        total_iou  += compute_iou(out, y)

    n = len(loader)
    return total_loss / n, total_iou / n


# ── Initialize model, optimizer, scheduler ───────────
model     = UNet(in_ch=CONFIG['in_channels'], out_ch=CONFIG['out_channels']).to(device)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                        weight_decay=CONFIG['weight_decay'])
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=CONFIG['patience_lr'], factor=0.5)

print(f"🚀 Starting training for {CONFIG['num_epochs']} epochs on {device}")
print(f"   LR: {CONFIG['learning_rate']:.0e} | Batch: {CONFIG['batch_size']}")
print(f"   Early stopping patience: {CONFIG['patience_es']}")
print("─" * 80)

# ── Training history ─────────────────────────────────
history = {
    'train_loss': [], 'val_loss': [],
    'train_iou': [],  'val_iou': [],
    'lr': [],
}

best_val_iou = 0.0
es_counter   = 0
best_weights = None
start_time   = time.time()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    epoch_start = time.time()

    # Train
    trn_loss, trn_iou = train_one_epoch(model, train_loader, optimizer, device)

    # Validate
    val_loss, val_iou = validate(model, val_loader, device)

    # Scheduler step
    scheduler.step(val_iou)
    current_lr = optimizer.param_groups[0]['lr']

    # Record history
    history['train_loss'].append(trn_loss)
    history['val_loss'].append(val_loss)
    history['train_iou'].append(trn_iou)
    history['val_iou'].append(val_iou)
    history['lr'].append(current_lr)

    # Early stopping check
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        es_counter = 0
        marker = " ⭐ BEST"
    else:
        es_counter += 1
        marker = f" (ES: {es_counter}/{CONFIG['patience_es']})"

    elapsed = time.time() - start_time
    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch:3d}/{CONFIG['num_epochs']} │ "
          f"TrnLoss: {trn_loss:.4f}  TrnIoU: {trn_iou:.4f} │ "
          f"ValLoss: {val_loss:.4f}  ValIoU: {val_iou:.4f} │ "
          f"LR: {current_lr:.1e} │ {epoch_time:.1f}s{marker}")

    if es_counter >= CONFIG['patience_es']:
        print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
        break

total_time = time.time() - start_time
print(f"\n✅ Training complete in {total_time/60:.1f} minutes")
print(f"   Best Val IoU: {best_val_iou:.4f}")


# %% [10] ── Training Curves ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

epochs_range = range(1, len(history['train_loss']) + 1)

# Loss curve
axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0].plot(epochs_range, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# IoU curve
axes[1].plot(epochs_range, history['train_iou'], 'b-', label='Train IoU', linewidth=2)
axes[1].plot(epochs_range, history['val_iou'], 'r-', label='Val IoU', linewidth=2)
axes[1].axhline(y=best_val_iou, color='g', linestyle='--', alpha=0.7, label=f'Best Val IoU: {best_val_iou:.4f}')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('IoU')
axes[1].set_title('Training & Validation IoU'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Learning rate curve
axes[2].plot(epochs_range, history['lr'], 'g-', linewidth=2)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Learning Rate Schedule'); axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Training curves saved to Google Drive")


# %% [11] ── Test Set Evaluation ─────────────────────────────────────────────

# Load best weights
model.load_state_dict(best_weights)
model.eval()

test_loss, test_iou = validate(model, test_loader, device)
print(f"📊 Test Set Results:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test IoU:  {test_iou:.4f}")

# ── Visualize sample predictions ─────────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(f'Sample Predictions (Test IoU: {test_iou:.4f})', fontsize=14)

with torch.no_grad():
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    preds = model(x_batch).cpu()

for i in range(min(3, x_batch.shape[0])):
    # RGB composite (bands 0,1,2)
    rgb = x_batch[i, :3].cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0, 1)

    # Ground truth mask
    gt = y_batch[i, 0].numpy()

    # Predicted mask
    pred = (preds[i, 0].numpy() > 0.5).astype(float)

    # Overlay
    overlay = rgb.copy()
    overlay[pred > 0.5] = [0, 1, 0]  # Green overlay for predicted RTH
    overlay = 0.6 * rgb + 0.4 * overlay

    axes[i, 0].imshow(rgb); axes[i, 0].set_title('RGB')
    axes[i, 1].imshow(gt, cmap='Greens'); axes[i, 1].set_title('Ground Truth')
    axes[i, 2].imshow(pred, cmap='Greens'); axes[i, 2].set_title('Prediction')
    axes[i, 3].imshow(overlay); axes[i, 3].set_title('Overlay')

    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
plt.show()


# %% [12] ── Save Model to Google Drive ──────────────────────────────────────
# This saves the model in the exact format expected by the Streamlit dashboard

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Save as active_model.pth (direct state_dict — same format as dashboard)
save_path = os.path.join(MODEL_OUTPUT_DIR, "active_model.pth")
torch.save(best_weights, save_path)
print(f"✅ Model saved to: {save_path}")

# Also save with descriptive name for versioning
version_name = f"unet_rth_ep{len(history['train_loss'])}_iou{best_val_iou:.3f}.pth"
version_path = os.path.join(MODEL_OUTPUT_DIR, version_name)
torch.save(best_weights, version_path)
print(f"✅ Versioned copy: {version_path}")

# Save training history
import json
history_path = os.path.join(MODEL_OUTPUT_DIR, "training_history.json")
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)
print(f"📊 Training history saved to: {history_path}")

print(f"\n{'='*60}")
print(f"  🎉 DONE! Next steps:")
print(f"  1. Download '{version_name}' from Google Drive")
print(f"  2. Open your Streamlit dashboard")
print(f"  3. Go to '🧠 Model & Pelatihan' → '📂 Unggah Model Pre-Trained'")
print(f"  4. Upload the .pth file")
print(f"{'='*60}")


# %% [13] ── (Optional) Download model directly ─────────────────────────────
# Uncomment to download the model file directly from Colab

# from google.colab import files
# files.download(version_path)
