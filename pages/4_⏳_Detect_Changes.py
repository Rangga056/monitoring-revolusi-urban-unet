"""
pages/4_⏳_Deteksi_Perubahan.py
Temporal change detection 2019 vs 2023.
Runs U-Net inference on Sentinel-2 GeoTIFF images and compares RTH coverage.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from utils import setup_page, DEFAULT_CONFIG, PROJECT_ROOT

setup_page("Change Detection", "⏳")

st.markdown('<h1 style="color:white;">⏳ GOS Change Detection (2019 vs 2023)</h1>', unsafe_allow_html=True)
st.markdown("""
This stage applies the U-Net model to the Jakarta imagery from **2019** and **2023** to detect  
temporal changes in GOS area using the downloaded Sentinel-2 data.
""")
st.divider()

# ── Path Configuration ───────────────────────────────────────────
TIF_2019 = os.path.join(PROJECT_ROOT, "data", "sentinel2_jakarta_2019.tif")
TIF_2023 = os.path.join(PROJECT_ROOT, "data", "sentinel2_jakarta_2023.tif")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "visualisasi")
CSV_PATH = os.path.join(RESULTS_DIR, "temporal_statistics.csv")

# ── Check active model ──────────────────────────────────────────
models_dir = os.path.join(PROJECT_ROOT, "models")
os.makedirs(models_dir, exist_ok=True)
available_models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

# Determine default option based on whether we have multiple models
if len(available_models) > 1:
    model_options = ["-- Ensemble (All Available Models) --"] + available_models
    default_index = 0
else:
    model_options = available_models
    default_index = 0

selected_option = st.selectbox("Choose model(s) for change detection:", model_options, index=default_index) if available_models else None

if selected_option and selected_option.startswith("-- Ensemble"):
    active_models = [os.path.join(models_dir, m) for m in available_models]
    st.success(f"✅ Ensemble mode active using **{len(active_models)}** models.")
elif selected_option:
    active_models = [os.path.join(models_dir, selected_option)]
    st.success(f"✅ Single model active: `{selected_option}`")
else:
    active_models = []

# ── Check data availability ──────────────────────────────────────
tif_2019_exists = os.path.exists(TIF_2019)
tif_2023_exists = os.path.exists(TIF_2023)
models_exist    = len(active_models) > 0

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Sentinel-2 2019", "✅ Available" if tif_2019_exists else "❌ Not available")
col_s2.metric("Sentinel-2 2023", "✅ Available" if tif_2023_exists else "❌ Not available")
col_s3.metric("U-Net Model(s)", "✅ Available" if models_exist else "❌ Not trained yet")

if not tif_2019_exists or not tif_2023_exists:
    st.warning(
        "⚠️ Sentinel-2 GeoTIFF files are not available yet.\n\n"
        f"- 2019: `{TIF_2019}`\n"
        f"- 2023: `{TIF_2023}`\n\n"
        "Run `python download_sentinel2.py` to download the data."
    )

if not models_exist:
    st.warning(
        "⚠️ U-Net Model is not available. Train the model on the **🧠 Model & Training** page "
        "or upload a `.pth` file first."
    )

st.divider()


# ── Inference function ───────────────────────────────────────────
def run_temporal_inference(active_models):
    """Run U-Net inference on full Sentinel-2 GeoTIFF to compute RTH statistics."""
    try:
        import torch
        import rasterio
        from utils import UNet
    except ImportError as e:
        st.error(f"❌ Dependencies not installed: {e}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load all models for ensemble or single model
    models = []
    for mp in active_models:
        m = UNet(in_ch=5, out_ch=1).to(device)
        m.load_state_dict(torch.load(mp, map_location=device))
        m.eval()
        models.append(m)

    patch_size = DEFAULT_CONFIG['patch_size']
    results = {}

    for year, tif_path in [("2019", TIF_2019), ("2023", TIF_2023)]:
        if not os.path.exists(tif_path):
            st.warning(f"⚠️ File {tif_path} not found, skipping.")
            continue

        st.write(f"📂 Reading image {year}...")
        with rasterio.open(tif_path) as src:
            img = src.read().astype(np.float32)  # (C, H, W)
            crs = src.crs
            transform = src.transform

            # Calculate pixel area in m²
            # GEE exports in EPSG:4326 (degrees), need to convert
            if crs and crs.is_geographic:
                # Convert degrees to meters at image center latitude
                center_lat = (src.bounds.top + src.bounds.bottom) / 2.0
                lat_rad = np.radians(abs(center_lat))
                # 1 degree latitude ≈ 111,320 m; longitude varies with cos(lat)
                m_per_deg_lat = 111320.0
                m_per_deg_lon = 111320.0 * np.cos(lat_rad)
                pixel_h_m = abs(src.res[1]) * m_per_deg_lat   # res[1] = y (lat)
                pixel_w_m = abs(src.res[0]) * m_per_deg_lon   # res[0] = x (lon)
                pixel_area_m2 = pixel_h_m * pixel_w_m
                st.write(f"   📐 CRS: {crs} (geographic) → pixel ≈ {pixel_h_m:.1f}×{pixel_w_m:.1f} m = {pixel_area_m2:.0f} m²")
            else:
                # Projected CRS: res is already in meters
                pixel_area_m2 = abs(src.res[0] * src.res[1])
                st.write(f"   📐 CRS: {crs} (projected) → pixel area = {pixel_area_m2:.0f} m²")

        # Ensure 5 bands (R, G, B, NIR, NDVI)
        if img.shape[0] == 4:
            nir = img[3]
            red = img[0]
            ndvi = (nir - red) / (nir + red + 1e-8)
            ndvi = np.clip(ndvi, -1.0, 1.0)
            img = np.concatenate([img, ndvi[np.newaxis]], axis=0)
        elif img.shape[0] >= 5:
            # NDVI already present as 5th band
            pass

        C, H, W = img.shape

        # Diagnostic: check value ranges to match normalization to training
        band_max = img[:4].max()
        band_min = img[:4].min()
        ndvi_max = img[4].max()
        ndvi_min = img[4].min()
        st.write(f"   📊 Band 0-3 range: [{band_min:.2f}, {band_max:.2f}]")
        st.write(f"   📊 NDVI range: [{ndvi_min:.4f}, {ndvi_max:.4f}]")

        # Normalize (same as RTHDataset in utils.py)
        # RTHDataset expects raw patches: bands 0-3 in 0-10000 range, NDVI in -1 to 1
        img_norm = img.copy()
        if band_max <= 1.5:
            # GEE toFloat() may output reflectance as 0-1 instead of 0-10000
            # In this case, skip the /10000 normalization for RGB/NIR
            st.write(f"   ⚠️ Bands 0-3 are already in the 0-1 range (GEE float). Using values directly.")
            img_norm[:4] = img[:4]
        else:
            img_norm[:4] = img_norm[:4] / 10000.0
        img_norm[4] = (np.clip(img_norm[4], -1.0, 1.0) + 1.0) / 2.0

        # Predict in patches
        pred_mask = np.zeros((H, W), dtype=np.float32)
        count_mask = np.zeros((H, W), dtype=np.float32)

        step = patch_size // 2  # 50% overlap for averaging
        total_patches = ((H - patch_size) // step + 1) * ((W - patch_size) // step + 1)
        progress = st.progress(0, text=f"Inference {year}: 0/{total_patches} patches")
        patch_i = 0

        with torch.no_grad():
            for y in range(0, H - patch_size + 1, step):
                for x in range(0, W - patch_size + 1, step):
                    patch = img_norm[:, y:y+patch_size, x:x+patch_size]
                    tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                    outs = [m(tensor).squeeze().cpu().numpy() for m in models]
                    out = np.mean(outs, axis=0)
                    pred_mask[y:y+patch_size, x:x+patch_size] += out
                    count_mask[y:y+patch_size, x:x+patch_size] += 1.0
                    patch_i += 1
                    if patch_i % 20 == 0:
                        progress.progress(
                            min(patch_i / max(total_patches, 1), 1.0),
                            text=f"Inference {year}: {patch_i}/{total_patches} patches"
                        )

        progress.progress(1.0, text=f"Inference {year}: complete ✅")

        # Average overlapping predictions
        count_mask[count_mask == 0] = 1
        pred_mask /= count_mask

        # Threshold
        binary_mask = (pred_mask > 0.5).astype(np.uint8)
        rth_pixels = int(binary_mask.sum())
        total_pixels = H * W
        rth_area_km2 = rth_pixels * pixel_area_m2 / 1e6  # m² → km²
        total_area_km2 = total_pixels * pixel_area_m2 / 1e6
        rth_pct = (rth_pixels / total_pixels) * 100

        results[year] = {
            "rth_pixels": rth_pixels,
            "total_pixels": total_pixels,
            "rth_area_km2": rth_area_km2,
            "total_area_km2": total_area_km2,
            "rth_pct": rth_pct,
            "binary_mask": binary_mask,
        }

        st.write(f"   ✅ {year}: GOS = {rth_area_km2:.2f} km² ({rth_pct:.2f}%)")

    if len(results) == 2:
        # Save statistics CSV
        os.makedirs(RESULTS_DIR, exist_ok=True)
        save_df = pd.DataFrame([
            {"Period": "2019", "GOS Area (km²)": results["2019"]["rth_area_km2"],
             "% GOS": results["2019"]["rth_pct"],
             "Total Area (km²)": results["2019"]["total_area_km2"]},
            {"Period": "2023", "GOS Area (km²)": results["2023"]["rth_area_km2"],
             "% GOS": results["2023"]["rth_pct"],
             "Total Area (km²)": results["2023"]["total_area_km2"]},
        ])
        save_df.to_csv(CSV_PATH, index=False)
        st.write(f"📊 Statistics saved to: `{CSV_PATH}`")

        # Generate change detection map
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            mask_2019 = results["2019"]["binary_mask"]
            mask_2023 = results["2023"]["binary_mask"]

            # Align sizes (use minimum)
            min_h = min(mask_2019.shape[0], mask_2023.shape[0])
            min_w = min(mask_2019.shape[1], mask_2023.shape[1])
            m19 = mask_2019[:min_h, :min_w]
            m23 = mask_2023[:min_h, :min_w]

            # Change map
            change = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            stayed = (m19 == 1) & (m23 == 1)
            gained = (m19 == 0) & (m23 == 1)
            lost   = (m19 == 1) & (m23 == 0)

            change[stayed] = [46, 204, 113]    # green  – GOS remained
            change[gained] = [241, 196, 15]    # yellow – New GOS
            change[lost]   = [231, 76, 60]     # red    – GOS lost

            map_path = os.path.join(RESULTS_DIR, "change_detection_map.png")
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            fig.patch.set_facecolor('#0e1117')
            ax.imshow(change)
            ax.set_title("GOS Change Detection Map 2019→2023", color='white', fontsize=14)
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(map_path, dpi=150, bbox_inches='tight',
                        facecolor='#0e1117', edgecolor='none')
            plt.close()
            st.write(f"🗺️ Change map saved to: `{map_path}`")
        except Exception as e:
            st.warning(f"⚠️ Failed to create change map: {e}")

        # Store in session state so results survive the rerun
        st.session_state['temporal_results'] = {
            'val_2019': results["2019"]["rth_area_km2"],
            'val_2023': results["2023"]["rth_area_km2"],
            'pr_2019': results["2019"]["rth_pct"],
            'pr_2023': results["2023"]["rth_pct"],
        }

    return results


# ── Determine if we have results to show ─────────────────────────
val_2019 = val_2023 = pr_2019 = pr_2023 = None
has_results = False

# Source 1: Session state (immediately after inference)
if 'temporal_results' in st.session_state:
    tr = st.session_state['temporal_results']
    val_2019 = tr['val_2019']
    val_2023 = tr['val_2023']
    pr_2019  = tr['pr_2019']
    pr_2023  = tr['pr_2023']
    has_results = True

# ── Actions ─────────────────────────────────────────
if tif_2019_exists and tif_2023_exists and models_exist:
    st.markdown("### 🚀 Run Temporal Analysis")
    
    col_inf, col_pres = st.columns(2)
    with col_inf:
        run_btn = st.button("🔍 Run Full Change Detection", type="primary", help="Run U-Net inference on the whole satellite image.")
        
    pres_btn = False
    with col_pres:
        if os.path.exists(CSV_PATH):
            pres_btn = st.button("⚡ Load Latest Presentation Data (Cached)", type="secondary", help="Instantly load pre-calculated statistics and maps.")

    if pres_btn:
        try:
            stats_df = pd.read_csv(CSV_PATH)
            
            # Determine column names based on English vs Indonesian
            col_period = 'Period' if 'Period' in stats_df.columns else 'Periode'
            col_area   = 'GOS Area (km²)' if 'GOS Area (km²)' in stats_df.columns else 'Luas RTH (km²)'
            col_pct    = '% GOS' if '% GOS' in stats_df.columns else '% RTH'

            # Safely extract values without Out of Bounds exception
            val_2019 = stats_df.loc[stats_df[col_period].astype(str).str.contains('2019'), col_area]
            val_2019 = float(val_2019.values[0]) if not val_2019.empty else 0.0
            
            val_2023 = stats_df.loc[stats_df[col_period].astype(str).str.contains('2023'), col_area]
            val_2023 = float(val_2023.values[0]) if not val_2023.empty else 0.0

            pr_2019 = stats_df.loc[stats_df[col_period].astype(str).str.contains('2019'), col_pct]
            pr_2019 = float(pr_2019.values[0]) if not pr_2019.empty else 0.0

            pr_2023 = stats_df.loc[stats_df[col_period].astype(str).str.contains('2023'), col_pct]
            pr_2023 = float(pr_2023.values[0]) if not pr_2023.empty else 0.0

            if val_2019 == 0.0 and val_2023 == 0.0:
                raise ValueError("Could not find rows matching 2019 or 2023 in the CSV.")

            st.session_state['temporal_results'] = {
                'val_2019': val_2019,
                'val_2023': val_2023,
                'pr_2019': pr_2019,
                'pr_2023': pr_2023,
            }
            has_results = True
            st.success("✅ Presentation data loaded instantly from cache!")
        except Exception as e:
            st.error(f"Failed to load cached presentation data: {e}. Please run a full Change Detection once.")

    elif run_btn:
        results = run_temporal_inference(active_models)
        if results and len(results) == 2:
            has_results = True
            val_2019 = st.session_state['temporal_results']['val_2019']
            val_2023 = st.session_state['temporal_results']['val_2023']
            pr_2019  = st.session_state['temporal_results']['pr_2019']
            pr_2023  = st.session_state['temporal_results']['pr_2023']
            st.success("✅ Inference complete! Results are displayed below.")

    if has_results:
        st.markdown("---")

# ── Display Results ──────────────────────────────────────────────
if not has_results:
    st.info("No analysis results yet. Run change detection above or ensure data and model are available.")
    st.stop()

# ── Key Metrics ──────────────────────────────────────────────────
perubahan = val_2023 - val_2019
pct_change = abs(perubahan) / val_2019 * 100 if val_2019 > 0 else 0

st.markdown("### 📊 Multitemporal GOS Area Statistics")
c1, c2, c3 = st.columns(3)
c1.metric("GOS Area 2019", f"{val_2019:.2f} km²", f"{pr_2019:.2f}% city area")
c2.metric("GOS Area 2023", f"{val_2023:.2f} km²", f"{pr_2023:.2f}% city area")
c3.metric("Net Change", f"{perubahan:.2f} km²",
          f"{'−' if perubahan < 0 else '+'}{pct_change:.1f}% over 4 years",
          delta_color="inverse" if perubahan < 0 else "normal")

st.markdown("---")

# ── Interactive Charts ───────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Area Bar Chart", "📉 Percentage Trend", "🗺️ Change Map"])

with tab1:
    df_bar = pd.DataFrame({
        "Year": ["2019", "2023"],
        "GOS Area (km²)": [val_2019, val_2023]
    })
    fig_bar = px.bar(df_bar, x='Year', y='GOS Area (km²)',
                     color='Year', text_auto='.2f',
                     color_discrete_map={"2019": "#2ecc71", "2023": "#e74c3c"},
                     title="DKI Jakarta GOS Area Comparison (km²)")
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white", showlegend=False,
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    df_pct = pd.DataFrame({
        "Year": ["2019", "2023"],
        "% GOS of City Area": [pr_2019, pr_2023]
    })
    fig_pct = go.Figure()
    fig_pct.add_trace(go.Scatter(
        x=df_pct["Year"], y=df_pct["% GOS of City Area"],
        mode='lines+markers+text',
        marker=dict(size=14, color=['#2ecc71', '#e74c3c']),
        line=dict(color='#95a5a6', width=2, dash='dot'),
        text=[f"{v:.2f}%" for v in df_pct["% GOS of City Area"]],
        textposition="top center"
    ))
    fig_pct.update_layout(
        title="GOS Percentage Trend Relative to City Area",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="% GOS Area"))
    st.plotly_chart(fig_pct, use_container_width=True)

with tab3:
    map_vis = os.path.join(RESULTS_DIR, "change_detection_map.png")
    if os.path.exists(map_vis):
        from PIL import Image
        st.image(Image.open(map_vis), use_container_width=True,
                 caption="GOS Change Detection Map 2019→2023")
        st.markdown("""
        **Legend:**  
        🟩 Green – GOS remained between 2019 and 2023  
        🟨 Yellow – New GOS (2023)  
        🟥 Red – GOS lost (land conversion)
        """)
    else:
        st.info(
            "The change map will be created automatically when you run the change detection above."
        )

# ── Conclusion callout ───────────────────────────────────────────
direction = "Reduction" if perubahan < 0 else "Increase"
st.markdown(f"""
<div style="background:rgba({'231,76,60' if perubahan < 0 else '46,204,113'},.12); border-left:4px solid {'#e74c3c' if perubahan < 0 else '#2ecc71'}; padding:16px; border-radius:6px; margin-top:20px;">
  <h4 style="margin-top:0; color:{'#e74c3c' if perubahan < 0 else '#2ecc71'}; font-size:1.1rem;">Temporal Analysis Conclusion</h4>
  A {direction.lower()} of approximately <strong>{pct_change:.1f}%</strong> ({abs(perubahan):.2f} km²) in GOS area over a period of <strong>4 years</strong>  
  {'indicates uncompensated conversion of vegetation land into built-up areas,' if perubahan < 0 else 'indicates an increase in green open space,'}
  serving as an important signal for the <em>Smart City</em> spatial planning policymakers of DKI Jakarta.
</div>
""", unsafe_allow_html=True)

# ── Raw Statistics Table ─────────────────────────────────────────
csv_stats_df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None
if csv_stats_df is not None:
    st.markdown("---")
    st.markdown("### 📋 Full Statistics Table")
    st.dataframe(csv_stats_df, use_container_width=True)
