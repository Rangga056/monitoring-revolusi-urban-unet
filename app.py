import streamlit as st
import pandas as pd
from utils import setup_page

# Set up the main page configuration
setup_page("Green Open Space Monitoring", "🛰️")

# Main Page Design
st.markdown('<h1 style="font-size: 2.8rem; background: -webkit-linear-gradient(45deg, #1cb5e0, #000046); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Green Open Space Revolution Monitoring</h1>', unsafe_allow_html=True)
st.markdown('<div style="font-size: 1.2rem; color: #a0aec0; margin-bottom: 30px;">A Deep Learning Approach Using U-Net Architecture on Sentinel-2 Satellite Imagery</div>', unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Background
    This research develops a semantic segmentation model based on the **U-Net** architecture to map and monitor Green Open Spaces (GOS) in Special Capital Region of Jakarta using **Sentinel-2** satellite imagery. 
    
    A multi-temporal approach (2019–2023) is used to analyze the dynamics of GOS changes within the context of smart city development (*smart environment*).
    
    The provision of adequate and evenly distributed GOS is a key indicator of a successful *smart environment*. Therefore, an accurate, fast, and automated monitoring tool is needed, as embodied in this dashboard.
    """)
    
with col2:
    st.info("""
    **Dashboard Objectives:**
    1. **Data Exploration:** Analysis of image availability and *ground truth* distribution.
    2. **Model Training:** Interactive module to train the U-Net artificial neural network.
    3. **Dynamic Evaluation:** Independent testing of IoU and F1-Score metrics.
    4. **Change Detection:** Multi-temporal GOS area map and analysis of percentage reduction.
    """)
    
st.markdown("### Methodology Summary")
method_data = pd.DataFrame({
    "Component": ["Satellite Data", "Ground Truth", "Model", "Input", "Loss Function", "Optimizer", "Evaluation"],
    "Details": [
        "Sentinel-2 Level-2A (Surface Reflectance), 10m resolution",
        "OpenStreetMap (OSM) land use polygons",
        "U-Net (4-level encoder-decoder) with Dropout",
        "5 channels: R, G, B, NIR, NDVI",
        "BCE + Dice Loss (combined) for Class Imbalance",
        "AdamW with ReduceLROnPlateau",
        "IoU, F1-Score, Precision, Recall, Accuracy, Kappa"
    ]
})
st.table(method_data)

st.markdown("### 🔄 End-to-End Process Flow")
import streamlit.components.v1 as components
    
mermaid_html = """
<div class="mermaid" style="display: flex; justify-content: center; background-color: transparent;">
flowchart TD
    A[Google Earth Engine Sentinel-2 L2A] --> C[Image Preprocessing - Cloud Mask, Median Composite]
    B[OpenStreetMap Ground Truth] --> E[Target Rasterization - Polygon to Binary Mask]
    
    C --> D[Feature Extraction - Calculate NDVI]
    D --> F[Patch Cropping - 256x256 px, 50% Overlap]
    E --> F
    
    F --> G[Split Dataset - Train 70%, Val 15%, Test 15%]
    G --> H[Data Augmentation - Flip, Rotate, Brightness, Noise]
    H --> I[U-Net Training - BCE + Dice Loss]
    
    I --> J[Model Evaluation - IoU, F1-Score, CM]
    J --> K[Export Trained Model - active_model.pth]
    
    K --> L[Change Detection]
    L --> M[Inference 2019 vs 2023 - GOS Area Difference]

    style A fill:#2b313e,stroke:#4a5568,color:#fff
    style B fill:#2b313e,stroke:#4a5568,color:#fff
    style C fill:#1e3a8a,stroke:#3b82f6,color:#fff
    style D fill:#1e3a8a,stroke:#3b82f6,color:#fff
    style E fill:#1e3a8a,stroke:#3b82f6,color:#fff
    style F fill:#1e3a8a,stroke:#3b82f6,color:#fff
    style G fill:#1e3a8a,stroke:#3b82f6,color:#fff
    style H fill:#1e3a8a,stroke:#3b82f6,color:#fff
    style I fill:#047857,stroke:#10b981,color:#fff
    style J fill:#b45309,stroke:#f59e0b,color:#fff
    style K fill:#047857,stroke:#10b981,color:#fff
    style L fill:#7e22ce,stroke:#a855f7,color:#fff
    style M fill:#7e22ce,stroke:#a855f7,color:#fff
</div>
<script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true });
</script>
"""
components.html(mermaid_html, height=750, scrolling=True)


st.markdown("""
---
*Use the navigation menu on the left to explore the dataset, perform U-Net model training processes, and view the interactive Green Open Space change analysis results.*
""")
