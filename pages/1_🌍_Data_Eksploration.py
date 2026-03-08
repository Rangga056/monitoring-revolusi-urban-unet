import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from utils import setup_page, DEFAULT_CONFIG

setup_page("Data & Exploration", "🌍")

st.markdown('<h1 style="color:white;">🌍 Data Acquisition & Exploration</h1>', unsafe_allow_html=True)
st.markdown("""
This page displays data sources, satellite image acquisition, and the distribution of field observations (*ground truth*) for Green Open Space (GOS) extracted from **OpenStreetMap (OSM)** polygons.  
All statistics are pulled directly from the local dataset in the `data/` folder.
""")
st.divider()

# ────────────────────────────────────────────────
# Section 1: Sentinel-2 info
# ────────────────────────────────────────────────
st.markdown("### 🛰️ Sentinel-2 Level-2A Image Acquisition")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
- **Source**: Google Earth Engine – `COPERNICUS/S2_SR_HARMONIZED`
- **Resolution**: 10 m (R, G, B, NIR Bands)
- **Input bands**: R · G · B · NIR · **NDVI** (5 channels)
- **Cloud masking**: QA60 bitflag (threshold 10 %)
- **Period**: 2019-01-01 – 2019-12-31 (T1), 2023-01-01 – 2023-12-31 (T2)
""")
with c2:
    st.info("""
**Normalization during training:**
- Band 0-3 (R/G/B/NIR): `÷ 10 000`  
- Band 4 (NDVI): `clamp(-1,1) → (val+1)/2`
""")

# ────────────────────────────────────────────────
# Section 2: OSM Ground Truth
# ────────────────────────────────────────────────
st.markdown("### 🗺️ Ground Truth – OSM GOS Polygons")

shp_path = DEFAULT_CONFIG['rth_shp']

if os.path.exists(shp_path):
    try:
        import geopandas as gpd

        @st.cache_data(show_spinner="Loading OSM shapefile...")
        def load_shp(path):
            gdf = gpd.read_file(path)
            gdf_proj = gdf.to_crs(epsg=32748)
            gdf['area_m2'] = gdf_proj.geometry.area
            gdf['area_ha'] = gdf['area_m2'] / 10_000
            return gdf

        gdf = load_shp(shp_path)
        class_stats = (gdf.groupby('fclass')
                         .agg(jumlah=('fclass', 'count'), luas_ha=('area_ha', 'sum'))
                         .sort_values('luas_ha', ascending=False)
                         .reset_index())

        total_poligon = int(class_stats['jumlah'].sum())
        total_luas    = class_stats['luas_ha'].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Poligon RTH</div>
                <div class="metric-value">{total_poligon:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">Total Luas RTH</div>
                <div class="metric-value blue">{total_luas:,.1f} Ha</div>
            </div>
            """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Area (Ha)", "🔢 Polygon Count", "📋 Full Table"])

        with tab1:
            fig = px.bar(class_stats, x='fclass', y='luas_ha',
                         color='fclass', text_auto='.1f',
                         labels={'fclass': 'GOS Class', 'luas_ha': 'Area (Ha)'},
                         title="GOS Area Distribution by Class (OSM)",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = px.pie(class_stats, values='jumlah', names='fclass',
                          title="Polygon Count Proportion",
                          hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Set3)
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               font_color="white")
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.dataframe(class_stats.rename(columns={
                'fclass': 'Class', 'jumlah': 'Polygon Count', 'luas_ha': 'Area (Ha)'}),
                use_container_width=True)

    except ImportError:
        st.warning("📦 `geopandas` is not installed. Displaying static notebook summary data.")

        def _show_static_stats():
            """Fallback: show hardcoded stats when geopandas is unavailable."""
            st.markdown("""
            | GOS Class       | Polygon Count | Area (Ha) |
            |-----------------|:--------------:|:---------:|
            | park            | ~1 200         | ~3 500    |
            | garden          | ~800           | ~1 200    |
            | grass           | ~600           | ~900      |
            | recreation      | ~300           | ~450      |
            | **Total**       | **~2 900**     | **~6 050**|

            > ⚠️ The data above is a static estimate. Install `geopandas` and provide the shapefile
            > to see accurate interactive statistics.
            """)

        _show_static_stats()

else:
    st.warning(f"Shapefile not found at `{shp_path}`. Make sure the `data/data-revolusi-urban/` folder exists.")

st.info("""
💡 **Class Imbalance:** GOS pixels only account for **~4.39 %** of the total area.  
The 1:21 ratio (GOS : Non-GOS) is handled with **BCE + Dice Loss** on the U-Net model.
""")

# ────────────────────────────────────────────────
# Section 3: Patch dataset statistics
# ────────────────────────────────────────────────
st.markdown("### 🔲 Patch Dataset Statistics (`Dataset_UNet_v2`)")

img_dir  = os.path.join(DEFAULT_CONFIG['dataset_dir'], 'images')
mask_dir = os.path.join(DEFAULT_CONFIG['dataset_dir'], 'masks')

if os.path.isdir(img_dir):
    n_patches = len([f for f in os.listdir(img_dir) if f.endswith('.npy')])
    n_train = int(n_patches * DEFAULT_CONFIG['split_ratios'][0])
    n_val   = int(n_patches * DEFAULT_CONFIG['split_ratios'][1])
    n_test  = n_patches - n_train - n_val

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patches", f"{n_patches:,}")
    c2.metric("Train (70 %)", f"{n_train:,}")
    c3.metric("Val (15 %)", f"{n_val:,}")
    c4.metric("Test (15 %)", f"{n_test:,}")

    st.markdown(f"""
    Patches are created with a size of `{DEFAULT_CONFIG['patch_size']}×{DEFAULT_CONFIG['patch_size']}` px  
    using a **sliding window** with an *overlap* of {DEFAULT_CONFIG['overlap']} px (50%).
    """)

    # Show a random sample patch
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.npy')])
    if img_files:
        import matplotlib.pyplot as plt

        sample_idx = st.slider("View Sample Patch", 0, len(img_files)-1, 0)
        sample_path = os.path.join(img_dir, img_files[sample_idx])
        mask_path   = os.path.join(mask_dir, img_files[sample_idx])

        img_data  = np.load(sample_path).astype(np.float32)
        mask_data = np.load(mask_path) if os.path.exists(mask_path) else None

        # Visualise RGB (bands 0,1,2 → normalise for display)
        rgb = img_data[[0,1,2], :, :].transpose(1, 2, 0)
        rgb = np.clip(rgb / rgb.max(), 0, 1) if rgb.max() > 0 else rgb

        fig, axes = plt.subplots(1, 2 if mask_data is not None else 1, figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117')
        if mask_data is not None:
            axes[0].imshow(rgb); axes[0].set_title("RGB Patch", color='white'); axes[0].axis('off')
            axes[1].imshow(mask_data, cmap='Greens'); axes[1].set_title("GOS Mask", color='white'); axes[1].axis('off')
        else:
            axes.imshow(rgb); axes.set_title("RGB Patch", color='white'); axes.axis('off')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
else:
    st.warning(f"Patch directory not found at `{img_dir}`.")

# ────────────────────────────────────────────────────────────────
# Section 4: Patch Quality Analysis
# ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 🔬 Patch Quality & GOS Distribution Analysis")

st.info("""
**The patch creation pipeline utilizes a hybrid strategy (OSM + NDVI):**  
- `all_touched=True` ensures small polygons are not lost during rasterization.
- NDVI threshold captures informal vegetation not registered in OSM.
- Patches with GOS density < 0.5% are automatically filtered out.
- Patches with GOS density ≥ 10% are oversampled (flip augmentation).
""")

sel_img_dir  = os.path.join(DEFAULT_CONFIG['dataset_dir'], "images")
sel_mask_dir = os.path.join(DEFAULT_CONFIG['dataset_dir'], "masks")

if os.path.isdir(sel_img_dir):
    mask_files = sorted([f for f in os.listdir(sel_img_dir) if f.endswith('.npy')])
    n_total = len(mask_files)

    if n_total == 0:
        st.warning("No patch files found.")
    else:
        sample_n = min(300, n_total)
        step     = max(1, n_total // sample_n)
        sample_files = mask_files[::step][:sample_n]

        rth_fracs = []
        for fname in sample_files:
            mp = os.path.join(sel_mask_dir, fname)
            if os.path.exists(mp):
                m = np.load(mp)
                rth_fracs.append(float(m.mean()))

        rth_fracs = np.array(rth_fracs)
        zero_pct  = (rth_fracs < 0.001).mean() * 100
        low_pct   = ((rth_fracs >= 0.001) & (rth_fracs < 0.05)).mean() * 100
        high_pct  = (rth_fracs >= 0.05).mean() * 100

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Total Patches", f"{n_total:,}")
        q2.metric("Empty (< 0.1% GOS)", f"{zero_pct:.1f}%",
                  delta="issue" if zero_pct > 50 else "ok", delta_color="inverse")
        q3.metric("Low GOS (0.1–5%)", f"{low_pct:.1f}%")
        q4.metric("High GOS (≥ 5%)", f"{high_pct:.1f}%",
                  delta="good" if high_pct > 10 else "low")

        fig_hist = px.histogram(
            x=rth_fracs * 100, nbins=50,
            labels={"x": "% GOS Pixels per Patch"},
            title=f"GOS Density Distribution (sample {sample_n:,} patches)",
            color_discrete_sequence=["#2ecc71"]
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(title="% GOS Pixels", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Number of Patches", gridcolor="rgba(255,255,255,0.1)"),
        )
        fig_hist.add_vline(x=0.5, line_dash="dash", line_color="#e74c3c",
                           annotation_text="min_rth_fraction (0.5%)",
                           annotation_font_color="#e74c3c")
        st.plotly_chart(fig_hist, use_container_width=True)

        if zero_pct > 50:
            st.error(
                f"⚠️ **{zero_pct:.0f}% patches are almost empty!** "
                "The model might tend to ignore vegetation.")
        elif zero_pct > 20:
            st.warning(f"⚠️ {zero_pct:.0f}% patches have very minimal GOS.")
        else:
            st.success(f"✅ GOS distribution is quite good — {high_pct:.1f}% patches have ≥ 5% GOS.")
else:
    st.warning(f"Dataset directory not found at `{sel_img_dir}`.")
