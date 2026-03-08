"""
pages/2_🧠_Model_Pelatihan.py
Training / Upload pre-trained model page.
Uses the REAL Dataset_UNet_v2 patch dataset (same pipeline as notebook).
Default hyperparameters match notebook CONFIG exactly.
"""
import streamlit as st
import time
import io
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
# Removed: from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import albumentations as A

from utils import (setup_page, get_device, set_seed, DEFAULT_CONFIG,
                   UNet, bce_dice_loss, RTHDataset, get_dataset_paths,
                   require_torch, TORCH_AVAILABLE, PROJECT_ROOT)

setup_page("Model & Training", "🧠")
require_torch()  # stops with install instructions if torch is missing


# ── Header ─────────────────────────────────────
st.markdown('<h1 style="color:white;">🧠 U-Net Architecture & Training Process</h1>', unsafe_allow_html=True)
st.markdown("""
Choose between training a new model using the local dataset `Dataset_UNet_v2_improved/`  
or uploading `.pth` weights from a previous training session.
""")
# ── Device badge ────────────────────────────────
device = get_device()
st.markdown(f"**Active Device:** `{'🟢 GPU – ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else '🔵 CPU'}`")
st.divider()

tab_train, tab_upload = st.tabs(["🚀 Train New Model", "📂 Upload Pre-Trained Model"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 – TRAIN
# ══════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("""
This page explains the model architecture, provides configuration options for training, 
and displays the training progress. The U-Net model is configured for **5-channel inputs**.
""")
    st.markdown("### ⚙️ Hyperparameter Configuration")
    st.caption("Default values are taken directly from the research notebook CONFIG.")

    c1, c2, c3 = st.columns(3)
    with c1:
        epochs        = st.number_input("Number of Epochs",  min_value=1,  max_value=100, value=DEFAULT_CONFIG['num_epochs'],    step=5)
        batch_size    = st.number_input("Batch Size",      min_value=1,  max_value=32,  value=DEFAULT_CONFIG['batch_size'])
    with c2:
        lr            = st.select_slider("Learning Rate", options=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                                          value=DEFAULT_CONFIG['learning_rate'],
                                          format_func=lambda x: f"{x:.0e}")
        weight_decay  = st.select_slider("Weight Decay",  options=[1e-3, 1e-4, 1e-5],
                                          value=DEFAULT_CONFIG['weight_decay'],
                                          format_func=lambda x: f"{x:.0e}")
    with c3:
        patience_es   = st.number_input("Early Stopping Patience", min_value=1, max_value=30, value=DEFAULT_CONFIG['patience_es'])
        patience_lr   = st.number_input("LR Scheduler Patience",   min_value=1, max_value=20, value=DEFAULT_CONFIG['patience_lr'])

    st.markdown("---")
    st.markdown("### 🧬 Data Augmentation")
    use_hflip  = st.checkbox("Horizontal Flip",           value=True)
    use_vflip  = st.checkbox("Vertical Flip",             value=True)
    use_rot90  = st.checkbox("Random Rotate 90°",          value=True)
    use_bright = st.checkbox("Random Brightness/Contrast", value=True)
    use_noise  = st.checkbox("Gaussian Noise",             value=True)

    st.markdown("---")

    # Check dataset existence first
    data_dir = DEFAULT_CONFIG['dataset_dir']
    json_path = os.path.join(data_dir, "spatial_folds.json")
    has_split = os.path.exists(json_path)

    if not os.path.exists(os.path.join(data_dir, "images")) or not os.path.exists(os.path.join(data_dir, "masks")):
        st.error(f"❌ Dataset not found at `{data_dir}`. "
                 f"Make sure the `images/` and `masks/` folders exist with `.npy` files.")
        st.stop()
    else:
        # Count total files by listing one of the directories
        total_files = len(os.listdir(os.path.join(data_dir, "images")))
        st.success(f"✅ Dataset found: **{total_files:,} patches** ready to use.")

    st.markdown("---")
    st.markdown("### 🗺️ Spatial Cross-Validation Fold Selection")

    # Fold selection for Spatial Cross Validation
    if has_split:
        with open(json_path, 'r') as f:
            folds_data = json.load(f)
        
        fold_options = ["-- Auto Train All Folds (4 Models) --"] + [f"Fold {k} - {v}" for k, v in folds_data['fold_desc'].items()]
        selected_fold_str = st.selectbox(
            "Select Validation Fold (Spatial Cross-Validation)", 
            options=fold_options,
            help="To prevent data leakage, testing should be done on a geographic area completely unseen during training. Selecting 'Auto Train All Folds' will run 4 sequential trainings."
        )
        if selected_fold_str.startswith("-- Auto"):
            val_fold_ids = [0, 1, 2, 3] # Training all 4 spatial folds
        else:
            val_fold_ids = [int(selected_fold_str.split()[1])] # Extracts 0, 1, 2, or 3
    else:
        st.warning("⚠️ No `spatial_folds.json` found! Please run `python prepare_patches_improved.py` to generate spatially stratified datasets before training.")
        st.stop()
    
    # --- START TRAINING ---
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        set_seed(DEFAULT_CONFIG['seed'])

        # ── Build augmentation pipeline ─────────────────────────
        aug_list = []
        if use_hflip:  aug_list.append(A.HorizontalFlip(p=0.5))
        if use_vflip:  aug_list.append(A.VerticalFlip(p=0.5))
        if use_rot90:  aug_list.append(A.RandomRotate90(p=0.5))
        if use_bright: aug_list.append(A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3))
        if use_noise:  aug_list.append(A.GaussNoise(p=0.2))
        train_aug = A.Compose(aug_list) if aug_list else None

        img_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")
        results_dir = os.path.join(PROJECT_ROOT, "results")
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Loop through all selected folds
        for current_fold_id in val_fold_ids:
            # --- Setup for current fold ---
            st.subheader(f"🚀 Training Fold {current_fold_id}")
            fold_container = st.container() # Create a container for fold-specific output
            with fold_container:
                status_text = st.empty()
                progress_bar = st.progress(0)
                chart_ph = st.empty() # Placeholder for the chart

                # Initialize model, optimizer, scheduler for each fold
                model = UNet(in_ch=5, out_ch=1).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience_lr)

                # Load fold data
                train_files = []
                val_files = []

                for patch in folds_data['folds']:
                    if patch['fold'] == current_fold_id:
                        val_files.append(patch['filename'])
                    else:
                        train_files.append(patch['filename'])

                # We need full absolute paths 
                train_img_paths = [os.path.join(img_dir, f) for f in train_files]
                train_mask_paths = [os.path.join(mask_dir, f) for f in train_files]

                val_img_paths = [os.path.join(img_dir, f) for f in val_files]
                val_mask_paths = [os.path.join(mask_dir, f) for f in val_files]
            
                st.write(f"📊 Training on {len(train_files)} patches (Folds != {current_fold_id})")
                st.write(f"📊 Validating on {len(val_files)} patches (Fold {current_fold_id})")

                train_dataset = RTHDataset(train_img_paths, train_mask_paths, transform=train_aug)
                val_dataset   = RTHDataset(val_img_paths,   val_mask_paths,   transform=None)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

                n_train_batches = len(train_loader)
                n_val_batches = len(val_loader)
                total_batches = n_train_batches * epochs + n_val_batches * epochs # Approximate total for progress bar

                best_val_iou = -1.0
                es_counter = 0
                best_weights = None
                global_batch = 0 # For overall progress bar
                chart_df = pd.DataFrame(columns=["Train Loss", "Val Loss", "Train IoU", "Val IoU"])

                def compute_iou(pred, target, thresh=0.5):
                    pred_bin = (pred > thresh).float()
                    inter = (pred_bin * target).sum()
                    union = pred_bin.sum() + target.sum() - inter
                    return (inter / (union + 1e-6)).item()

                start = time.time()

                for epoch in range(1, epochs + 1):
                    # ── Train ──
                    model.train()
                    trn_loss, trn_iou = 0.0, 0.0
                    for batch_i, (x, y) in enumerate(train_loader, 1):
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()
                        out  = model(x)
                        loss = bce_dice_loss(out, y)
                        loss.backward()
                        optimizer.step()
                        trn_loss += loss.item()
                        trn_iou  += compute_iou(out.detach(), y)
                        global_batch += 1

                        # Update progress every batch
                        elapsed = time.time() - start
                        pct = global_batch / total_batches
                        eta = (elapsed / global_batch) * (total_batches - global_batch) if global_batch > 0 else 0
                        progress_bar.progress(min(pct, 1.0),
                            text=f"Epoch {epoch}/{epochs} | Train batch {batch_i}/{n_train_batches} | "
                                 f"⏱️ {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
                        status_text.markdown(
                            f"**Epoch {epoch}/{epochs}** — Training batch `{batch_i}/{n_train_batches}` | "
                            f"Loss: `{loss.item():.4f}` | Elapsed: `{elapsed/60:.1f} min`"
                        )

                    trn_loss /= len(train_loader); trn_iou /= len(train_loader)

                    # ── Val ──
                    model.eval()
                    val_loss, val_iou = 0.0, 0.0
                    with torch.no_grad():
                        for x, y in val_loader:
                            x, y = x.to(device), y.to(device)
                            out  = model(x)
                            val_loss += bce_dice_loss(out, y).item()
                            val_iou  += compute_iou(out, y)
                    val_loss /= len(val_loader); val_iou /= len(val_loader)

                    # ── Update UI (end of epoch) ──
                    elapsed = time.time() - start
                    
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    new_row = pd.DataFrame([{
                        "Train Loss": trn_loss, "Val Loss": val_loss,
                        "Train IoU": trn_iou,   "Val IoU": val_iou
                    }])
                    chart_df = pd.concat([chart_df, new_row], ignore_index=True)

                    ax1.plot(chart_df.index+1, chart_df["Train Loss"], color='tab:red',   label='Train Loss', marker='o')
                    ax1.plot(chart_df.index+1, chart_df["Val Loss"],   color='tab:orange',label='Val Loss',   marker='o')
                    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (BCE+Dice)')
                    
                    ax2 = ax1.twinx()
                    ax2.plot(chart_df.index+1, chart_df["Train IoU"],  color='tab:blue',  label='Train IoU',  marker='s')
                    ax2.plot(chart_df.index+1, chart_df["Val IoU"],    color='tab:cyan',  label='Val IoU',    marker='s')
                    ax2.set_ylabel('IoU Score')
                    
                    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
                    chart_ph.pyplot(fig)
                    plt.close(fig)
                    
                    status_text.success(
                        f"**Epoch {epoch}/{epochs} completed!** ⏱️ `{elapsed/60:.1f}m`\n"
                        f"- Train Loss: `{trn_loss:.4f}` | Train IoU: `{trn_iou:.4f}`\n"
                        f"- Val Loss: `{val_loss:.4f}` | Val IoU: `{val_iou:.4f}`"
                    )

                    scheduler.step(val_iou)

                    # ── Early stopping ──
                    if val_iou > best_val_iou:
                        best_val_iou = val_iou
                        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        es_counter = 0
                    else:
                        es_counter += 1
                        if es_counter >= patience_es:
                            status_text.warning(f"⏹️ Early stopping triggered at epoch {epoch}.")
                            break

                # --- 💾 Save & Download for this specific fold ---
                # Always save dynamically based on the fold ID so we can use Ensemble mode later.
                chart_name = f"training_curves_fold_{current_fold_id}.png"
                model_name = f"active_model_fold_{current_fold_id}.pth"

                # Save plot
                fig.savefig(os.path.join(results_dir, chart_name), dpi=150, bbox_inches='tight')
                
                # Save Model
                st.success(f"🎉 Training Fold {current_fold_id} completed! Best Val IoU: **{best_val_iou:.4f}**")
                save_path = os.path.join(models_dir, model_name)
                torch.save(best_weights, save_path)
                st.info(f"Model physically saved to: `{save_path}`")

                # In-memory buffer for the download button
                buf = io.BytesIO()
                torch.save(best_weights, buf)
                buf.seek(0)
                st.download_button(
                    label=f"⬇️ Download Fold {current_fold_id} Weights (.pth)",
                    data=buf.getvalue(),
                    file_name=f"unet_rth_fold{current_fold_id}_ep{epochs}_iou{best_val_iou:.3f}.pth",
                    mime="application/octet-stream",
                    key=f"download_btn_fold_{current_fold_id}" # Must be unique across iterations
                )
            st.session_state['model_ready'] = True


# ══════════════════════════════════════════════════════════════════
# TAB 2 – UPLOAD
# ══════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### 📂 Upload Trained PyTorch Models (.pth)")
    st.markdown("Upload your trained `.pth` files (from Colab or a previous session) so you don't need to retrain.")
    st.markdown("You can upload **multiple models** simultaneously to enable Ensemble mode in Evaluation pages.")

    uploaded_files = st.file_uploader("Select model files (.pth)", type=["pth"], accept_multiple_files=True)
    if uploaded_files:
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        success_count = 0
        for uploaded in uploaded_files:
            try:
                # Test load the weights to verify it's a valid PyTorch dict
                state_dict = torch.load(uploaded, map_location=device)
                
                # Verify UNet architecture compatibility
                model = UNet(in_ch=5, out_ch=1).to(device)
                model.load_state_dict(state_dict)
                model.eval()

                # Save physically using the original uploaded filename
                save_path = os.path.join(models_dir, uploaded.name)
                
                # Reset file pointer and save explicitly from the memory buffer 
                # or just use the state dict to ensure integrity.
                torch.save(state_dict, save_path)
                success_count += 1
                
                st.success(f"✅ Model `{uploaded.name}` successfully uploaded and saved.")

            except Exception as e:
                st.error(f"❌ Failed to load model `{uploaded.name}`: {e}")
                
        if success_count > 0:
            st.info(f"🎉 **{success_count} model(s)** ready! You can now open the **Model Evaluation** or **Change Detection** pages.")
            st.session_state['model_ready'] = True
