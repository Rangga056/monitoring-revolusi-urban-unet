"""
utils.py - Shared utilities for the RTH Monitoring Streamlit Dashboard
Matches exact U-Net architecture and data pipeline from the original notebook.
torch is imported lazily so the dashboard can start even before torch is installed.
"""

import os
import random
import numpy as np
import streamlit as st

# ── Project root (absolute path to this file's directory) ────────────────────
# This ensures data paths always resolve correctly regardless of Streamlit CWD.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Optional heavy imports ────────────────────────────────────────────────────
# torch is only required on pages that train/infer. The home/data pages will
# work without it; pages that need it will check TORCH_AVAILABLE themselves.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None       # type: ignore
    nn    = None       # type: ignore
    F     = None       # type: ignore
    Dataset = object   # fallback base class so RTHDataset can still be defined

# ============================================================
# DEFAULT CONFIG (ported exactly from notebook Tahap 2)
# ============================================================
DEFAULT_CONFIG = {
    'roi_name': 'DKI Jakarta',
    'bbox': [106.68, -6.37, 106.98, -6.08],
    # Actual paths: data folder contains a same-name subfolder (nested structure)
    'dataset_dir': os.path.join(PROJECT_ROOT, 'data', 'Dataset_UNet_v2_improved'),
    'rth_shp':     os.path.join(PROJECT_ROOT, 'data', 'data-revolusi-urban', 'data-revolusi-urban', 'jakarta_rth_filtered.shp'),
    'patch_size': 256,
    'overlap': 128,
    'in_channels': 5,
    'out_channels': 1,
    'batch_size': 8,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience_lr': 5,
    'patience_es': 10,
    'split_ratios': (0.70, 0.15, 0.15),
    'seed': 42,
    'rth_classes': ['forest', 'grass', 'park', 'meadow', 'recreation_ground',
                    'nature_reserve', 'village_green', 'garden', 'orchard'],
}

# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================
# DEVICE
# ============================================================
def get_device():
    if not TORCH_AVAILABLE:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# U-NET ARCHITECTURE – matches notebook Tahap 9 exactly
# If torch is not installed the classes are stubs; pages that need them
# should check TORCH_AVAILABLE before instantiating.
# ============================================================
if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        def forward(self, x): return self.conv(x)

    class UNet(nn.Module):
        def __init__(self, in_ch=5, out_ch=1):
            super().__init__()
            self.enc1 = DoubleConv(in_ch, 64)
            self.enc2 = DoubleConv(64, 128)
            self.enc3 = DoubleConv(128, 256)
            self.enc4 = DoubleConv(256, 512)
            self.pool = nn.MaxPool2d(2, 2)
            self.bottleneck = DoubleConv(512, 1024)
            self.drop_bn = nn.Dropout2d(0.5)
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

    # ── Loss Functions (BCE + Dice – notebook Tahap 10) ──────────────────────
    def dice_loss(pred, target, smooth=1e-6):
        intersection = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        return 1. - ((2. * intersection + smooth) / (union + smooth)).mean()

    def bce_dice_loss(pred, target):
        bce = F.binary_cross_entropy(pred, target)
        return bce + dice_loss(pred, target)

    # ── Dataset (notebook Tahap 8) ────────────────────────────────────────────
    class RTHDataset(Dataset):
        """
        Loads .npy patch files from Dataset_UNet_v2/images and masks.
        Normalisation matches notebook exactly:
          Band 0-3: ÷10000   |  Band 4 (NDVI): clamp(-1,1) → (v+1)/2
        """
        def __init__(self, image_paths, mask_paths, transform=None):
            self.image_paths = image_paths
            self.mask_paths  = mask_paths
            self.transform   = transform

        def __len__(self): return len(self.image_paths)

        def __getitem__(self, idx):
            img  = np.load(self.image_paths[idx]).astype(np.float32)
            mask = np.load(self.mask_paths[idx]).astype(np.float32)
            img[:4]  = img[:4] / 10000.0
            img[4]   = (np.clip(img[4], -1.0, 1.0) + 1.0) / 2.0
            img_hwc  = np.transpose(img, (1, 2, 0))
            if self.transform:
                aug     = self.transform(image=img_hwc, mask=mask)
                img_hwc = aug['image']
                mask    = aug['mask']
            img_t  = torch.from_numpy(np.transpose(img_hwc, (2, 0, 1)).copy())
            mask_t = torch.from_numpy(mask.copy()).unsqueeze(0)
            return img_t, mask_t

else:
    # Stub classes so imports in pages don't crash at module level
    class DoubleConv:  pass    # type: ignore
    class UNet:        pass    # type: ignore
    class RTHDataset:  pass    # type: ignore
    def dice_loss(*a, **kw):   return None
    def bce_dice_loss(*a, **kw): return None

# ============================================================
# DATASET LOADING HELPERS (no torch needed)
# ============================================================
def get_dataset_paths(dataset_dir: str):
    """Returns sorted lists of image and mask .npy paths."""
    image_dir = os.path.join(dataset_dir, 'images')
    mask_dir  = os.path.join(dataset_dir, 'masks')
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        return [], []
    imgs  = sorted([os.path.join(image_dir, f)
                    for f in os.listdir(image_dir) if f.endswith('.npy')])
    masks = sorted([os.path.join(mask_dir, f)
                    for f in os.listdir(mask_dir) if f.endswith('.npy')])
    return imgs, masks

def load_model_from_file(filepath, device=None):
    """Load a UNet from a .pth weight file."""
    if not TORCH_AVAILABLE:
        return None
    if device is None:
        device = get_device()
    model = UNet(in_ch=5, out_ch=1).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    return model

# ============================================================
# STREAMLIT PAGE SETUP
# ============================================================
GLOBAL_CSS = """
<style>
    .metric-card {
        background: #1e2530;
        padding: 18px 22px;
        border-radius: 10px;
        border-left: 5px solid #2ecc71;
        box-shadow: 0 4px 8px rgba(0,0,0,.15);
        margin-bottom: 16px;
    }
    .metric-card.blue  { border-left-color: #3498db; }
    .metric-card.red   { border-left-color: #e74c3c; }
    .metric-card.gold  { border-left-color: #f39c12; }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #2ecc71; }
    .metric-value.blue { color: #3498db; }
    .metric-value.red  { color: #e74c3c; }
    .metric-value.gold { color: #f39c12; }
    .metric-label { font-size: 0.9rem; color: #e2e8f0; }
    div[data-testid="stMetric"] { background: #1e2530; border-radius: 10px; padding: 10px; }
    div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p, div[data-testid="stMetricLabel"] span { color: #a0aec0 !important; }
    div[data-testid="stMetricValue"] { color: #ffffff !important; }
</style>
"""

def setup_page(title: str, icon: str):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def require_torch():
    """Call at top of any page that needs torch. Shows error and stops if unavailable."""
    if not TORCH_AVAILABLE:
        st.error("""
        ❌ **PyTorch is not detected** in this environment.

        Please install it by running the following command in the terminal:
        ```bash
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install scikit-learn albumentations
        ```
        Then restart Streamlit with `streamlit run app.py`.
        """)
        st.stop()
