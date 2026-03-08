"""
pages/3_📈_Evaluasi_Model.py
Dynamic model evaluation on test set with a full prediction explorer.
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, cohen_kappa_score)

from utils import setup_page, get_device, DEFAULT_CONFIG, UNet, RTHDataset, get_dataset_paths, require_torch, PROJECT_ROOT

setup_page("Model Evaluation", "📈")
require_torch()

st.markdown('<h1 style="color:white;">📈 U-Net Model Performance Evaluation</h1>', unsafe_allow_html=True)
st.markdown("Evaluation is performed on a separate **Test Set** that was not used during training. "
            "Click the button to run inference and explore all predictions.")
st.divider()

# ── Check active model ──────────────────────────────────────────
models_dir = os.path.join(PROJECT_ROOT, "models")
available_models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

if not available_models:
    st.warning("⚠️ No active models found. Train or upload weights on the **Model & Training** page first.")
    st.stop()

# Determine default option based on whether we have multiple models
if len(available_models) > 1:
    model_options = ["-- Ensemble (All Available Models) --"] + available_models
    default_index = 0
else:
    model_options = available_models
    default_index = 0

selected_option = st.selectbox("Choose model(s) for evaluation:", model_options, index=default_index)

if selected_option.startswith("-- Ensemble"):
    active_models = [os.path.join(models_dir, m) for m in available_models]
    st.success(f"✅ Ensemble mode active using **{len(active_models)}** models.")
else:
    active_models = [os.path.join(models_dir, selected_option)]
    st.success(f"✅ Single model active: `{selected_option}`")

# ── Load datasets ───────────────────────────────────────────────
imgs, masks = get_dataset_paths(DEFAULT_CONFIG['dataset_dir'])
if not imgs:
    st.error(f"Dataset not found at `{DEFAULT_CONFIG['dataset_dir']}`.")
    st.stop()

_, temp_imgs, _, temp_masks = train_test_split(imgs, masks, test_size=0.30, random_state=DEFAULT_CONFIG['seed'])
_, test_imgs, _, test_masks = train_test_split(temp_imgs, temp_masks, test_size=0.50, random_state=DEFAULT_CONFIG['seed'])

# ── Sidebar params ──────────────────────────────────────────────
batch_size = st.sidebar.number_input("Batch Size (inference)", 1, 32, 8)
threshold  = st.sidebar.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.05,
                                help="Values above threshold → GOS prediction")

# ── Run evaluation ──────────────────────────────────────────────
results_dir = os.path.join(PROJECT_ROOT, "results")
os.makedirs(results_dir, exist_ok=True)
cache_path = os.path.join(results_dir, "evaluation_cache.npz")

col_inf, col_pres = st.columns(2)
with col_inf:
    run_btn = st.button("🔍 Run Evaluation on Test Set", type="primary")

pres_btn = False
with col_pres:
    if os.path.exists(cache_path):
        pres_btn = st.button("⚡ Load Latest Presentation Data (Cached)", type="secondary", 
                             help="Bypass inference and instantly load the last successful evaluation results.")

if pres_btn:
    try:
        data = np.load(cache_path, allow_pickle=True)
        st.session_state['eval_preds']     = data['eval_preds']
        st.session_state['eval_targets']   = data['eval_targets']
        st.session_state['eval_raw']       = list(data['eval_raw'])
        st.session_state['eval_threshold'] = float(data['eval_threshold'])
        st.session_state['test_imgs']      = data['test_imgs'].tolist()
        st.session_state['test_masks']     = data['test_masks'].tolist()
        st.success("✅ Presentation data loaded instantly from cache!")
    except Exception as e:
        st.error(f"Failed to load cache: {e}")

elif run_btn:
    device = get_device()
    
    # Load all models for ensemble or single model
    models = []
    for mp in active_models:
        m = UNet(in_ch=5, out_ch=1).to(device)
        m.load_state_dict(torch.load(mp, map_location=device))
        m.eval()
        models.append(m)

    test_ds     = RTHDataset(test_imgs, test_masks)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    all_preds_raw: list[np.ndarray] = []   # per-patch raw outputs for explorer
    prog = st.progress(0, text="Running inference...")

    import time
    start_time = time.time()
    total_batches = len(test_loader)
    
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            # Average predictions from all models in the ensemble
            outs = [m(x).cpu().numpy() for m in models]
            out = np.mean(outs, axis=0)           # (B, 1, H, W)
            for b in range(out.shape[0]):
                all_preds_raw.append(out[b, 0])   # save raw proba per patch
            pred = (out > threshold).astype(np.uint8).reshape(-1)
            tgt  = y.numpy().astype(np.uint8).reshape(-1)
            all_preds.extend(pred.tolist())
            all_targets.extend(tgt.tolist())
            
            # Progress calculation
            batch_num = i + 1
            elapsed = time.time() - start_time
            eta = (elapsed / batch_num) * (total_batches - batch_num) if batch_num > 0 else 0
            
            prog.progress(
                batch_num / total_batches, 
                text=f"Running inference: Batch {batch_num}/{total_batches} | ⏱️ {elapsed/60:.1f}m | ETA: {eta/60:.1f}m"
            )

    prog.progress(1.0, text=f"Inference complete! ⏱️ {time.time()-start_time:.1f}s")

    # Persist evaluations in session so tabs can use them
    st.session_state['eval_preds']     = np.array(all_preds)
    st.session_state['eval_targets']   = np.array(all_targets)
    st.session_state['eval_raw']       = all_preds_raw    # list[np.ndarray]
    st.session_state['eval_threshold'] = threshold
    st.session_state['test_imgs']      = test_imgs
    st.session_state['test_masks']     = test_masks

    # Save to cache for presentation mode
    np.savez_compressed(
        cache_path,
        eval_preds=np.array(all_preds),
        eval_targets=np.array(all_targets),
        eval_raw=np.array(all_preds_raw, dtype=object),
        eval_threshold=threshold,
        test_imgs=np.array(test_imgs),
        test_masks=np.array(test_masks)
    )
    st.success(f"💾 Results saved for future presentations!")

# ── Show results if available ────────────────────────────────────
if 'eval_preds' not in st.session_state:
    st.info("Click the button above to start evaluation.")
    st.stop()

all_preds   = st.session_state['eval_preds']
all_targets = st.session_state['eval_targets']
raw_outputs = st.session_state['eval_raw']
thr         = st.session_state['eval_threshold']

# ── Metrics ─────────────────────────────────────────────────────
acc   = accuracy_score(all_targets, all_preds)
prec  = precision_score(all_targets, all_preds, zero_division=0)
rec   = recall_score(all_targets, all_preds, zero_division=0)
f1    = f1_score(all_targets, all_preds, zero_division=0)
inter = ((all_preds == 1) & (all_targets == 1)).sum()
union = ((all_preds == 1) | (all_targets == 1)).sum()
iou   = inter / union if union > 0 else 0.0
kappa = cohen_kappa_score(all_targets, all_preds)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_metrics, tab_matrix, tab_explorer = st.tabs(
    ["📊 Metrics & Summary", "🧮 Confusion Matrix", "🔬 Full Prediction Exploration"])

with tab_metrics:
    st.markdown("### Metrics Summary (Test Set)")
    metrics = [("Accuracy", acc, ""), ("Precision", prec, "blue"),
               ("Recall", rec, ""), ("F1-Score", f1, "blue"),
               ("IoU", iou, "gold"), ("Cohen's Kappa", kappa, "gold")]
    cols = st.columns(3)
    for i, (name, val, color) in enumerate(metrics):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">{name}</div>
                <div class="metric-value {color}">{val:.4f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    df_summary = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "IoU", "Cohen's Kappa"],
        "Value":  [acc, prec, rec, f1, iou, kappa]
    })
    st.dataframe(df_summary.style.format({"Value": "{:.4f}"}), use_container_width=True)

    # ── Save Metrics to CSV ──────────────────────────────────────────
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_csv_path = os.path.join(results_dir, "evaluation_metrics.csv")
    metrics_dict = {
        "Accuracy": [acc], "Precision": [prec], "Recall": [rec], 
        "F1-Score": [f1], "IoU": [iou], "Cohen's Kappa": [kappa]
    }
    pd.DataFrame(metrics_dict).to_csv(metrics_csv_path, index=False)
    st.info(f"📊 Evaluation (CSV) saved to `results/evaluation_metrics.csv`")

with tab_matrix:
    st.markdown("### 🧮 Interactive Confusion Matrix")
    cm = confusion_matrix(all_targets, all_preds)
    fig_cm = ff.create_annotated_heatmap(
        cm.tolist(),
        x=["Pred Non-GOS", "Pred GOS"],
        y=["Actual Non-GOS", "Actual GOS"],
        colorscale="Viridis", showscale=True
    )
    fig_cm.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", margin=dict(t=50, l=80, r=50, b=50))
    st.plotly_chart(fig_cm, use_container_width=True)

    # ── Save Confusion Matrix High Res ──────────────────────────────────────────
    # Plotly requires Kaleido which sometimes errors, so generating matplotlib equivalent
    fig_mat, ax_mat = plt.subplots(figsize=(6, 5))
    cax = ax_mat.matshow(cm, cmap='viridis')
    fig_mat.colorbar(cax)
    for (i, j), z in np.ndenumerate(cm):
        ax_mat.text(j, i, f'{z}', ha='center', va='center', color='white' if z < (cm.max()/2) else 'black')
    ax_mat.set_xticklabels([''] + ["Pred Non-GOS", "Pred GOS"])
    ax_mat.set_yticklabels([''] + ["Actual Non-GOS", "Actual GOS"])
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    
    results_dir = os.path.join(PROJECT_ROOT, "results")
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig_mat)

    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positive",  f"{tp:,}")
    c2.metric("True Negative",  f"{tn:,}")
    c3.metric("False Positive", f"{fp:,}", delta=f"-{fp/(tp+fp)*100:.1f}% Precision" if (tp+fp) > 0 else "", delta_color="inverse")
    c4.metric("False Negative", f"{fn:,}", delta=f"-{fn/(tp+fn)*100:.1f}% Recall"  if (tp+fn) > 0 else "", delta_color="inverse")

# ── Tab 3: Full patch explorer ──────────────────────────────────
with tab_explorer:
    st.markdown("### 🔬 Explore All Prediction Results")
    st.markdown(
        f"There are **{len(raw_outputs):,} test patches** that you can explore below.  "
        "Use the slider or navigation buttons to select a patch."
    )

    test_imgs_ev  = st.session_state['test_imgs']
    test_masks_ev = st.session_state['test_masks']
    n_patches     = len(raw_outputs)

    # Items per page (grid)
    per_page = st.select_slider("Patches per page", options=[1, 2, 4, 6, 9, 12], value=6)
    total_pages = max(1, (n_patches + per_page - 1) // per_page)

    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    st.caption(f"Page {page} / {total_pages}  •  Total {n_patches} patches")

    start_idx = (page - 1) * per_page
    end_idx   = min(start_idx + per_page, n_patches)
    cols_per_row = min(per_page, 3)   # max 3 columns per row

    patch_indices = list(range(start_idx, end_idx))
    ncols = cols_per_row
    nrows = (len(patch_indices) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows * 2, ncols * 2, figsize=(4 * ncols * 2, 4 * nrows * 2))
    axes = np.array(axes).reshape(nrows * 2, ncols * 2)
    fig.patch.set_facecolor('#0e1117')
    fig.suptitle(f"GOS Segmentation Results – U-Net on Sentinel-2",
                 color='white', fontsize=15, y=1.01)

    for pi, idx in enumerate(patch_indices):
        row_block = (pi // ncols) * 2
        col_block = (pi %  ncols) * 2

        img_arr  = np.load(test_imgs_ev[idx]).astype(np.float32)
        mask_arr = np.load(test_masks_ev[idx]) if os.path.exists(test_masks_ev[idx]) else None
        pred_arr = (raw_outputs[idx] > thr).astype(np.uint8)

        # overlay: TP=green, FP=yellow, FN=red
        overlay = np.zeros((*pred_arr.shape, 3), dtype=np.float32)
        if mask_arr is not None:
            overlay[(mask_arr == 1) & (pred_arr == 1)] = [0, 1, 0]   # TP green
            overlay[(mask_arr == 0) & (pred_arr == 1)] = [1, 1, 0]   # FP yellow
            overlay[(mask_arr == 1) & (pred_arr == 0)] = [1, 0, 0]   # FN red

        rgb = img_arr[[0, 1, 2], :, :].transpose(1, 2, 0)
        rgb = np.clip(rgb / rgb.max(), 0, 1) if rgb.max() > 0 else rgb

        # Row 0: RGB + Ground Truth
        for ax in [axes[row_block, col_block], axes[row_block, col_block + 1]]:
            ax.set_facecolor('#0e1117')
        axes[row_block, col_block].imshow(rgb)
        axes[row_block, col_block].set_title(f"RGB Image", color='white', fontsize=9)
        axes[row_block, col_block].axis('off')
        if mask_arr is not None:
            axes[row_block, col_block + 1].imshow(mask_arr, cmap='Greens', vmin=0, vmax=1)
        axes[row_block, col_block + 1].set_title("Ground Truth", color='white', fontsize=9)
        axes[row_block, col_block + 1].axis('off')

        # Row 1: Prediction + Overlay
        for ax in [axes[row_block + 1, col_block], axes[row_block + 1, col_block + 1]]:
            ax.set_facecolor('#0e1117')
        axes[row_block + 1, col_block].imshow(pred_arr, cmap='Greens', vmin=0, vmax=1)
        axes[row_block + 1, col_block].set_title("U-Net Prediction", color='white', fontsize=9)
        axes[row_block + 1, col_block].axis('off')
        axes[row_block + 1, col_block + 1].imshow(overlay)
        axes[row_block + 1, col_block + 1].set_title("Overlay", color='white', fontsize=9)
        axes[row_block + 1, col_block + 1].axis('off')

    # Hide unused axes
    for pi2 in range(len(patch_indices), nrows * ncols):
        r = (pi2 // ncols) * 2; c = (pi2 % ncols) * 2
        for dr in range(2):
            for dc in range(2):
                if r + dr < axes.shape[0] and c + dc < axes.shape[1]:
                    axes[r + dr, c + dc].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ── Save Predictions High Res ──────────────────────────────────────────
    results_dir = os.path.join(PROJECT_ROOT, "results")
    predictions_path = os.path.join(results_dir, "sample_predictions.png")
    fig.savefig(predictions_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    
    st.markdown("---")
    
    # Enable exporting universally
    st.markdown("#### 💾 Export All Predictions")
    st.markdown("This feature will export **all patches (all pages)** into high-resolution `.png` image files into the `results/predictions/` folder.")
    if st.button("Save All Prediction Images"):
        export_prog = st.progress(0, text="Preparing image export...")
        export_dir = os.path.join(results_dir, "predictions")
        os.makedirs(export_dir, exist_ok=True)
        
        for p in range(1, total_pages + 1):
            export_prog.progress(p / total_pages, text=f"Saving page {p} of {total_pages}...")
            s_idx = (p - 1) * per_page
            e_idx = min(s_idx + per_page, n_patches)
            
            p_indices = list(range(s_idx, e_idx))
            p_nrows = (len(p_indices) + ncols - 1) // ncols
            
            fig_ex, axes_ex = plt.subplots(p_nrows * 2, ncols * 2, figsize=(4 * ncols * 2, 4 * p_nrows * 2))
            axes_ex = np.array(axes_ex).reshape(p_nrows * 2, ncols * 2)
            fig_ex.patch.set_facecolor('#0e1117')
            fig_ex.suptitle(f"GOS Segmentation Results – Page {p}", color='white', fontsize=15, y=1.01)

            for pi, idx in enumerate(p_indices):
                row_block = (pi // ncols) * 2
                col_block = (pi %  ncols) * 2

                img_arr  = np.load(test_imgs_ev[idx]).astype(np.float32)
                mask_arr = np.load(test_masks_ev[idx]) if os.path.exists(test_masks_ev[idx]) else None
                pred_arr = (raw_outputs[idx] > thr).astype(np.uint8)

                overlay = np.zeros((*pred_arr.shape, 3), dtype=np.float32)
                if mask_arr is not None:
                    overlay[(mask_arr == 1) & (pred_arr == 1)] = [0, 1, 0]
                    overlay[(mask_arr == 0) & (pred_arr == 1)] = [1, 1, 0]
                    overlay[(mask_arr == 1) & (pred_arr == 0)] = [1, 0, 0]

                rgb = img_arr[[0, 1, 2], :, :].transpose(1, 2, 0)
                rgb = np.clip(rgb / rgb.max(), 0, 1) if rgb.max() > 0 else rgb

                for ax in [axes_ex[row_block, col_block], axes_ex[row_block, col_block + 1]]:
                    ax.set_facecolor('#0e1117')
                axes_ex[row_block, col_block].imshow(rgb)
                axes_ex[row_block, col_block].set_title(f"RGB Image", color='white', fontsize=9)
                axes_ex[row_block, col_block].axis('off')
                if mask_arr is not None:
                    axes_ex[row_block, col_block + 1].imshow(mask_arr, cmap='Greens', vmin=0, vmax=1)
                axes_ex[row_block, col_block + 1].set_title("Ground Truth", color='white', fontsize=9)
                axes_ex[row_block, col_block + 1].axis('off')

                for ax in [axes_ex[row_block + 1, col_block], axes_ex[row_block + 1, col_block + 1]]:
                    ax.set_facecolor('#0e1117')
                axes_ex[row_block + 1, col_block].imshow(pred_arr, cmap='Greens', vmin=0, vmax=1)
                axes_ex[row_block + 1, col_block].set_title("U-Net Prediction", color='white', fontsize=9)
                axes_ex[row_block + 1, col_block].axis('off')
                axes_ex[row_block + 1, col_block + 1].imshow(overlay)
                axes_ex[row_block + 1, col_block + 1].set_title("Overlay", color='white', fontsize=9)
                axes_ex[row_block + 1, col_block + 1].axis('off')

            for pi2 in range(len(p_indices), p_nrows * ncols):
                r = (pi2 // ncols) * 2; c = (pi2 % ncols) * 2
                for dr in range(2):
                    for dc in range(2):
                        if r + dr < axes_ex.shape[0] and c + dc < axes_ex.shape[1]:
                            axes_ex[r + dr, c + dc].set_visible(False)

            fig_ex.tight_layout()
            out_path = os.path.join(export_dir, f"page_{p:03d}.png")
            fig_ex.savefig(out_path, dpi=300, bbox_inches='tight', facecolor=fig_ex.get_facecolor())
            plt.close(fig_ex)
            
        export_prog.empty()
        st.success(f"✅ All prediction images successfully saved to `{export_dir}`")

    # Legend
    st.markdown("""
    **Overlay Legend:**  
    🟩 **Green** = TP (Correct prediction: GOS)  
    🟨 **Yellow** = FP (False prediction: Non-GOS predicted as GOS)  
    🟥 **Red** = FN (False prediction: GOS not detected)
    """)
