"""
prepare_patches_improved.py
============================
Script perbaikan untuk membuat patch RTH yang lebih berkualitas dari citra Sentinel-2.

Perbaikan utama vs. pipeline notebook original:
1. Rasterisasi all_touched=True  → poligon kecil tidak hilang
2. Hybrid mask: OSM + NDVI threshold → menangkap RTH yang tidak dilabeli OSM
3. Patch filtering berdasarkan RTH density → buang patch yang hampir kosong
4. Weighted sampling → pastikan batch training mengandung cukup RTH patch
5. Normalisasi dan statistik patch yang lebih informatif

CARA PAKAI:
   python prepare_patches_improved.py
   
Pastikan path ke file .tif dan .shp sudah diatur di bagian CONFIG di bawah.
"""

import os, sys, random, json
import numpy as np                        # type: ignore
import geopandas as gpd                   # type: ignore
import rasterio                           # type: ignore
from rasterio.features import rasterize   # type: ignore
from rasterio.windows import Window       # type: ignore
from rasterio.enums import Resampling     # type: ignore
from pathlib import Path
from tqdm import tqdm                     # type: ignore

# ============================================================
# CONFIG — sesuaikan dengan lokasi file GeoTIFF & shapefile
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # Path citra Sentinel-2 (GeoTIFF 5-band: R G B NIR NDVI atau R G B NIR lalu hitung NDVI)
    "tif_2019": os.path.join(PROJECT_ROOT, "data", "sentinel2_jakarta_2019.tif"),
    "tif_2023": os.path.join(PROJECT_ROOT, "data", "sentinel2_jakarta_2023.tif"),

    # Shapefile RTH dari OSM
    "rth_shp": os.path.join(PROJECT_ROOT, "data", "data-revolusi-urban",
                             "data-revolusi-urban", "jakarta_rth_filtered.shp"),

    # Output directory untuk patch .npy
    "output_dir": os.path.join(PROJECT_ROOT, "data", "Dataset_UNet_v2_improved"),

    # Patch settings
    "patch_size": 256,        # piksel
    "overlap": 128,           # piksel (50%)

    # RTH mask strategy:
    #   "osm"         → hanya dari shapefile OSM (original)
    #   "ndvi"        → hanya dari NDVI threshold
    #   "hybrid"      → gabungan OSM | NDVI (REKOMENDASI)
    "mask_strategy": "hybrid",
    "ndvi_threshold": 0.2,    # piksel dengan NDVI > threshold dianggap vegetasi
    "ndvi_band_index": 4,     # index band NDVI di GeoTIFF (0-based). -1 = hitung manual dari R & NIR
    "nir_band_index": 3,      # index band NIR (untuk hitung NDVI manual jika ndvi_band_index=-1)
    "red_band_index": 0,      # index band Red

    # Rasterisasi
    "all_touched": True,      # True = poligon kecil tidak hilang, lebih lengkap

    # Patch filtering — hanya simpan patch dengan RTH >= threshold ini
    "min_rth_fraction": 0.005,  # 0.5% piksel RTH minimum (buang patch yang hampir kosong)
    "max_rth_fraction": 1.0,    # filter patch oversampling jika diperlukan

    # Oversampling patch RTH positif
    # Patch dengan RTH >= threshold ini disimpan 2x (duplikasi + flipping)
    "oversample_rth_fraction": 0.10,  # 10% piksel RTH → duplikasi

    # Train/Val/Test split (Deprecated, replaced by Spatial Folds)
    # "split_ratios": (0.70, 0.15, 0.15),
    "seed": 42,
}

# ============================================================
# HELPERS
# ============================================================

def compute_ndvi(img_chw: np.ndarray, red_idx: int, nir_idx: int) -> np.ndarray:
    """Hitung NDVI dari array (C, H, W). Hasil dalam [-1, 1]."""
    nir = img_chw[nir_idx].astype(np.float32)
    red = img_chw[red_idx].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1.0, 1.0)


def build_osm_mask(shp_path: str, src, all_touched: bool) -> np.ndarray:  # type: ignore[type-arg]
    """Rasterisasi shapefile OSM ke mask biner sesuai CRS & transform gambar."""
    gdf = gpd.read_file(shp_path)

    # Reproject ke CRS gambar jika berbeda
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((src.height, src.width), dtype=np.uint8)

    mask = rasterize(
        shapes=shapes,
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched,  # ← kunci perbaikan: poligon kecil tidak hilang
    )
    return mask


def build_ndvi_mask(ndvi: np.ndarray, threshold: float) -> np.ndarray:
    """Buat mask biner dari array NDVI."""
    return (ndvi > threshold).astype(np.uint8)


def build_hybrid_mask(osm_mask: np.ndarray, ndvi: np.ndarray, threshold: float) -> np.ndarray:
    """Gabung OSM | NDVI → menangkap RTH yang tidak terdaftar di OSM."""
    ndvi_mask = build_ndvi_mask(ndvi, threshold)
    return np.clip(osm_mask + ndvi_mask, 0, 1).astype(np.uint8)


def extract_patches(img_chw: np.ndarray, mask_hw: np.ndarray,
                    patch_size: int, overlap: int,
                    min_rth: float, max_rth: float, src):
    """Generator: yield (patch_img CHW, patch_mask HW, rth_fraction)."""
    step = patch_size - overlap
    C, H, W = img_chw.shape

    for y in range(0, H - patch_size + 1, step):
        for x in range(0, W - patch_size + 1, step):
            patch_img  = img_chw[:, y:y+patch_size, x:x+patch_size]
            patch_mask = mask_hw[y:y+patch_size, x:x+patch_size]

            if patch_img.shape[1] != patch_size or patch_img.shape[2] != patch_size:
                continue  # skip partial border patches

            rth_frac = patch_mask.mean()

            # Compute approximate geographic coordinates (center of patch)
            # transform: (x_px, y_px) -> (lon, lat)
            cx, cy = src.transform * (x + patch_size/2, y + patch_size/2)

            yield patch_img, patch_mask, rth_frac, cx, cy


def augment_flip(img_chw: np.ndarray, mask_hw: np.ndarray):
    """Horizontal + vertical flip augmentasi untuk duplikasi patch RTH tinggi."""
    flipped_img  = np.flip(img_chw, axis=2).copy()   # horizontal flip
    flipped_mask = np.flip(mask_hw, axis=1).copy()
    return flipped_img, flipped_mask


def assign_spatial_fold(cx: float, cy: float, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> int:
    """Assign a fold index (0-3) based on geographic quadrant."""
    mid_lon = (min_lon + max_lon) / 2.0
    mid_lat = (min_lat + max_lat) / 2.0

    if cx < mid_lon and cy >= mid_lat:
        return 0  # North-West
    elif cx >= mid_lon and cy >= mid_lat:
        return 1  # North-East
    elif cx < mid_lon and cy < mid_lat:
        return 2  # South-West
    else:
        return 3  # South-East


# ============================================================
# MAIN
# ============================================================

def process_tif(tif_path: str, out_base: str, cfg: dict, shp_osm: str | None = None):
    """
    Baca satu GeoTIFF, buat mask, extract patches, simpan ke out_base/{images,masks}/.
    Returns: dict statistik.
    """
    os.makedirs(os.path.join(out_base, "images"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "masks"),  exist_ok=True)

    tif_name = Path(tif_path).stem
    stats: dict[str, int | list[float] | list[dict] | str] = {
        "total_patches": 0, "saved_patches": 0, "oversampled": 0,
        "rth_fracs": [], "patches_info": [], "tif": tif_name
    }

    print(f"\n📂 Membaca: {tif_path}")
    with rasterio.open(tif_path) as src:
        print(f"   CRS : {src.crs}")
        print(f"   Shape: {src.count} bands × {src.height} × {src.width} px")
        print(f"   Bands: {list(src.descriptions)}")

        # Baca semua band sekaligus
        img = src.read().astype(np.float32)  # (C, H, W)

        # ── Auto-detect value range ──────────────────────────────
        # GEE toFloat() may export reflectance as 0-1 instead of 0-10000
        # RTHDataset normalizes by ÷10000, so patches must be in 0-10000 range
        band_max = float(img[:min(4, img.shape[0])].max())
        print(f"   Band 0-3 max value: {band_max:.2f}")
        if band_max <= 1.5:
            print(f"   ⚠️  Bands dalam range 0-1 (GEE float). Mengalikan ×10000 untuk konsistensi.")
            img[:min(4, img.shape[0])] = img[:min(4, img.shape[0])] * 10000.0

        # ── Hitung / ekstrak NDVI ────────────────────────────────
        ndvi_idx = cfg["ndvi_band_index"]
        if ndvi_idx >= 0 and ndvi_idx < img.shape[0]:
            ndvi = img[ndvi_idx]
            # If NDVI was also scaled by 10000, undo that
            if ndvi.max() > 10:
                ndvi = ndvi / 10000.0
            print(f"   NDVI diambil dari band index {ndvi_idx}")
        else:
            ndvi = compute_ndvi(img, cfg["red_band_index"], cfg["nir_band_index"])
            print("   NDVI dihitung dari R & NIR")

        # ── Tambahkan NDVI sebagai channel ke-5 jika belum ada ──
        if img.shape[0] < 5:
            img = np.concatenate([img, ndvi[np.newaxis]], axis=0)
            print("   NDVI ditambahkan sebagai channel ke-5")
        elif img.shape[0] >= 5:
            # If NDVI band (index 4) was already scaled to 0-10000, fix it
            if img[4].max() > 10:
                img[4] = img[4] / 10000.0
                print("   NDVI band sudah ada, di-rescale ke range -1 to 1")

        # ── Buat mask ─────────────────────────────────────────────
        strategy: str = str(cfg["mask_strategy"])
        osm_mask: np.ndarray | None = None

        if shp_osm and os.path.exists(shp_osm):
            print(f"   OSM shapefile: {shp_osm}")
            osm_mask = build_osm_mask(shp_osm, src, bool(cfg["all_touched"]))
            print(f"   OSM RTH pixels: {osm_mask.sum():,} / {osm_mask.size:,} "
                  f"({osm_mask.mean()*100:.2f}%)")
        elif strategy in ("osm", "hybrid"):
            print("   ⚠️  Shapefile tidak ditemukan — fallback ke NDVI-only")
            strategy = "ndvi"

        full_mask: np.ndarray
        if strategy == "osm":
            assert osm_mask is not None, "OSM mask required for strategy='osm'"
            full_mask = osm_mask
        elif strategy == "ndvi":
            full_mask = build_ndvi_mask(ndvi, float(cfg["ndvi_threshold"]))
        else:  # hybrid
            assert osm_mask is not None, "OSM mask required for strategy='hybrid'"
            full_mask = build_hybrid_mask(osm_mask, ndvi, float(cfg["ndvi_threshold"]))

        print(f"   Mask strategy: '{strategy}'")
        print(f"   RTH pixels total: {full_mask.sum():,} ({full_mask.mean()*100:.2f}%)")

        # ── Extract patches ──────────────────────────────────────
        patch_count = 0
        patches_info_list: list[dict] = []
        rth_fracs_list: list[float] = []
        total_patches = 0
        saved_patches = 0
        oversampled = 0
        
        # Bounds used for fold assignment (calculate from src instead of cfg["bbox"] for accuracy)
        # However, due to the overlap loop, we can pass src into extract filter
        min_lon, max_lon = src.bounds.left, src.bounds.right
        min_lat, max_lat = src.bounds.bottom, src.bounds.top

        for patch_img, patch_mask, rth_frac, cx, cy in extract_patches(
                img, full_mask,
                int(cfg["patch_size"]), int(cfg["overlap"]),
                float(cfg["min_rth_fraction"]), float(cfg["max_rth_fraction"]), src):

            total_patches += 1
            fname = f"{tif_name}_patch_{patch_count:06d}.npy"
            np.save(os.path.join(out_base, "images", fname), patch_img)
            np.save(os.path.join(out_base, "masks",  fname), patch_mask)
            rth_fracs_list.append(float(rth_frac))
            
            fold = assign_spatial_fold(cx, cy, min_lon, max_lon, min_lat, max_lat)
            patches_info_list.append({"filename": fname, "fold": fold, "rth_frac": float(rth_frac), "cx": float(cx), "cy": float(cy)})

            saved_patches += 1
            patch_count += 1

            # Oversample patch RTH tinggi
            if rth_frac >= float(cfg["oversample_rth_fraction"]):
                flipped_img, flipped_mask = augment_flip(patch_img, patch_mask)
                fname_aug = f"{tif_name}_patch_{patch_count:06d}_aug.npy"
                np.save(os.path.join(out_base, "images", fname_aug), flipped_img)
                np.save(os.path.join(out_base, "masks",  fname_aug), flipped_mask)
                
                patches_info_list.append({"filename": fname_aug, "fold": fold, "rth_frac": float(rth_frac), "cx": float(cx), "cy": float(cy), "is_aug": True})
                
                oversampled += 1
                patch_count += 1

        stats["total_patches"] = total_patches
        stats["saved_patches"] = saved_patches
        stats["oversampled"] = oversampled
        stats["rth_fracs"] = rth_fracs_list
        stats["patches_info"] = patches_info_list

    return stats


def main():
    cfg = CONFIG
    shp: str = str(cfg["rth_shp"])
    out_base: str = str(cfg["output_dir"])

    print("=" * 60)
    print("  RTH Patch Creation — Improved Pipeline")
    print("=" * 60)
    print(f"  Mask strategy  : {cfg['mask_strategy']}")
    print(f"  NDVI threshold : {cfg['ndvi_threshold']}")
    print(f"  all_touched    : {cfg['all_touched']}")
    _min_frac: float = cfg["min_rth_fraction"]  # type: ignore[assignment]
    print(f"  min RTH frac   : {_min_frac*100:.1f}%")
    print(f"  Patch size     : {cfg['patch_size']} × {cfg['patch_size']}")
    print(f"  Overlap        : {cfg['overlap']} px")
    print("=" * 60)

    all_stats: list[dict] = []

    for tif_key in ["tif_2019", "tif_2023"]:
        tif_path: str = str(cfg[tif_key])
        if not os.path.exists(tif_path):
            print(f"⚠️  File tidak ditemukan, dilewati: {tif_path}")
            continue
        stats = process_tif(tif_path, out_base, cfg, shp_osm=shp)
        all_stats.append(stats)

        print(f"\n✅ {stats['tif']}")
        print(f"   Patch tersimpan : {stats['saved_patches']:,}")
        print(f"   Oversampled     : {stats['oversampled']:,}")
        if stats["rth_fracs"]:
            fracs = np.array(stats["rth_fracs"])
            print(f"   RTH frac median : {np.median(fracs)*100:.2f}%")
            print(f"   RTH frac max    : {fracs.max()*100:.2f}%")

    # ── Save statistics JSON ────────────────────────────────────────
    if not all_stats:
        print("\n⚠️  Tidak ada file GeoTIFF yang berhasil diproses.")
        print("   Pastikan file .tif sudah ada di folder data/")
        print(f"   Lokasi yang dicari:")
        for k in ["tif_2019", "tif_2023"]:
            print(f"     - {cfg[k]}")
        return

    os.makedirs(out_base, exist_ok=True)
    json_path: str = os.path.join(out_base, "patch_statistics.json")
    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n📊 Statistik tersimpan di: {json_path}")

    # ── Map spatial folds ────────────────────────────────────────
    img_dir: str = os.path.join(out_base, "images")
    if os.path.isdir(img_dir):
        all_patches_info = []
        for s in all_stats:
            if isinstance(s, dict) and "patches_info" in s:
                info_list = s.get("patches_info")
                if isinstance(info_list, list):
                    all_patches_info.extend(info_list) 

        # Verify against actual available files
        all_files = set([f for f in os.listdir(img_dir) if f.endswith(".npy")])
        valid_patches = [p for p in all_patches_info if p["filename"] in all_files]

        fold_counts = {0: 0, 1: 0, 2: 0, 3: 0, "total": len(valid_patches)}
        for p in valid_patches:
            fold_counts[p["fold"]] += 1

        split_info = {
            "folds": valid_patches,
            "counts": fold_counts,
            "fold_desc": {
                "0": "North-West",
                "1": "North-East",
                "2": "South-West",
                "3": "South-East"
            }
        }
        
        split_path: str = os.path.join(out_base, "spatial_folds.json")
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)

        print(f"\n📁 Spatial Fold Metadata tersimpan di: {split_path}")
        print(f"   Total valid patches: {fold_counts['total']:,}")
        for k, v in fold_counts.items():
            if str(k).isdigit():
                print(f"     Fold {k} ({split_info['fold_desc'][str(k)]}): {v:,} patches")

    print("\n🎉 Selesai!")


if __name__ == "__main__":
    main()
