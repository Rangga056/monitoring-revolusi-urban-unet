"""
download_sentinel2.py
=====================
Download Sentinel-2 Level-2A composites from Google Earth Engine for DKI Jakarta.

Output:
  data/sentinel2_jakarta_2019.tif   (5-band: R, G, B, NIR, NDVI)
  data/sentinel2_jakarta_2023.tif   (5-band: R, G, B, NIR, NDVI)

CARA PAKAI:
  1. Install dependencies:        pip install earthengine-api geemap
  2. Autentikasi GEE (sekali):    earthengine authenticate
  3. Jalankan script:             python download_sentinel2.py

Catatan:
  - Script ini menggunakan `geemap` untuk download GeoTIFF langsung ke disk.
  - Resolusi: 10 m/px (native Sentinel-2 Band 2,3,4,8)
  - Cloud masking: QA60 bit-flag (cirrus + opaque cloud)
  - Composite: median tahunan (mengurangi noise dan cloud residual)
"""

import os
import sys

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Region of Interest: DKI Jakarta ───────────────────────────────────────────
# Bounding box [West, South, East, North]
BBOX = [106.68, -6.37, 106.98, -6.08]

# ── Year configurations ───────────────────────────────────────────────────────
YEARS = {
    2019: {
        "start": "2019-01-01",
        "end":   "2019-12-31",
        "output": os.path.join(DATA_DIR, "sentinel2_jakarta_2019.tif"),
    },
    2023: {
        "start": "2023-01-01",
        "end":   "2023-12-31",
        "output": os.path.join(DATA_DIR, "sentinel2_jakarta_2023.tif"),
    },
}

# ── Band configuration ────────────────────────────────────────────────────────
# Sentinel-2 L2A band names at 10 m resolution
S2_BANDS = ["B4", "B3", "B2", "B8"]  # Red, Green, Blue, NIR
OUTPUT_BAND_NAMES = ["Red", "Green", "Blue", "NIR", "NDVI"]
SCALE = 10  # metres per pixel

# ── Google Cloud Project ID ───────────────────────────────────────────────────
# GEE sekarang membutuhkan project ID.
# Cara mendapatkan:
#   1. Buka https://console.cloud.google.com/
#   2. Buat project baru (gratis) atau gunakan yang sudah ada
#   3. Aktifkan Earth Engine API: https://console.cloud.google.com/apis/library/earthengine.googleapis.com
#   4. Isi project ID di bawah, atau biarkan None untuk input interaktif
GEE_PROJECT = "pro-router-461108-b8"


def initialize_ee():
    """Initialize and authenticate Google Earth Engine."""
    try:
        import ee
    except ImportError:
        print("❌ 'earthengine-api' belum terinstall.")
        print("   Jalankan:  pip install earthengine-api geemap")
        sys.exit(1)

    # Determine project ID
    project = GEE_PROJECT

    # Try to initialize (auth may already exist)
    def _try_init(proj):
        if proj:
            ee.Initialize(project=proj)
        else:
            ee.Initialize()

    try:
        _try_init(project)
        print("✅ Google Earth Engine berhasil diinisialisasi.")
        return ee
    except Exception as init_err:
        # Check if it's a project error
        if "no project found" in str(init_err).lower() or "project" in str(init_err).lower():
            if not project:
                print("=" * 60)
                print("🔐 GEE membutuhkan Google Cloud Project ID.")
                print()
                print("Cara mendapatkan (gratis):")
                print("  1. Buka: https://console.cloud.google.com/")
                print("  2. Buat project baru atau pilih yang sudah ada")
                print("  3. Aktifkan Earth Engine API:")
                print("     https://console.cloud.google.com/apis/library/earthengine.googleapis.com")
                print("  4. Daftar Earth Engine: https://signup.earthengine.google.com/")
                print("=" * 60)
                project = input("\nMasukkan GEE Project ID: ").strip()
                if not project:
                    print("❌ Project ID diperlukan.")
                    sys.exit(1)
        else:
            # Needs authentication first
            print("🔐 Autentikasi GEE diperlukan...")
            try:
                ee.Authenticate()
            except Exception as auth_err:
                print(f"❌ Gagal autentikasi: {auth_err}")
                sys.exit(1)

            if not project:
                project = input("\nMasukkan GEE Project ID: ").strip()
                if not project:
                    print("❌ Project ID diperlukan.")
                    sys.exit(1)

    # Retry with project
    try:
        _try_init(project)
        print(f"✅ GEE berhasil diinisialisasi dengan project: {project}")
        print(f"   💡 Tip: set GEE_PROJECT = \"{project}\" di script agar tidak perlu input lagi.")
        return ee
    except Exception as e:
        print(f"❌ Gagal inisialisasi GEE: {e}")
        sys.exit(1)


def mask_s2_clouds(image, ee):
    """Mask clouds using Sentinel-2 QA60 band (bit 10 = cloud, bit 11 = cirrus)."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask)


def compute_ndvi(image, ee):
    """Add NDVI band computed from NIR (B8) and Red (B4)."""
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)


def get_s2_composite(ee, start_date, end_date, roi):
    """
    Build a cloud-free median composite from Sentinel-2 L2A.
    Returns an ee.Image with bands: Red, Green, Blue, NIR, NDVI (float32).
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    n_images = collection.size().getInfo()
    print(f"   Citra ditemukan: {n_images} scenes (cloud < 30%)")

    if n_images == 0:
        print("   ❌ Tidak ada citra yang memenuhi kriteria!")
        return None

    # Apply cloud mask & compute NDVI, then take median composite
    composite = (
        collection
        .map(lambda img: mask_s2_clouds(img, ee))
        .map(lambda img: compute_ndvi(img, ee))
        .select(S2_BANDS + ["NDVI"])
        .median()
        .rename(OUTPUT_BAND_NAMES)
        .clip(roi)
        .toFloat()
    )

    return composite


def download_geotiff(ee, image, roi, output_path, scale=10):
    """
    Export ee.Image to Google Drive, wait for completion, then download to local disk.
    This bypasses the 50 MB direct download limit.
    """
    import time

    filename = os.path.splitext(os.path.basename(output_path))[0]
    drive_folder = "GEE_Sentinel2_Export"

    print(f"   📤 Exporting ke Google Drive (folder: {drive_folder})...")
    print(f"      Nama file: {filename}.tif")

    # Start the export task
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=filename,
        folder=drive_folder,
        fileNamePrefix=filename,
        region=roi,
        scale=scale,
        maxPixels=1e9,
        fileFormat="GeoTIFF",
    )
    task.start()

    # Poll for completion
    print(f"   ⏳ Menunggu export selesai (cek status tiap 15 detik)...")
    while True:
        status = task.status()
        state = status.get("state", "UNKNOWN")

        if state == "COMPLETED":
            print(f"   ✅ Export selesai!")
            break
        elif state == "FAILED":
            error_msg = status.get("error_message", "Unknown error")
            print(f"   ❌ Export gagal: {error_msg}")
            return False
        elif state == "CANCELLED":
            print(f"   ❌ Export dibatalkan.")
            return False
        else:
            # READY, RUNNING, etc.
            print(f"      Status: {state}...", end="\r")
            time.sleep(15)

    # Download from Google Drive
    print(f"   📥 Downloading dari Google Drive ke: {output_path}")
    try:
        downloaded = _download_from_drive(filename + ".tif", drive_folder, output_path)
        if downloaded:
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   ✅ Berhasil! Ukuran: {size_mb:.1f} MB")
            return True
    except Exception as e:
        print(f"   ⚠️  Auto-download dari Drive gagal: {e}")

    print()
    print(f"   📁 File sudah ada di Google Drive Anda:")
    print(f"      Folder : My Drive/{drive_folder}/")
    print(f"      File   : {filename}.tif")
    print(f"   👉 Download manual dan simpan ke: {output_path}")
    return False


def _download_from_drive(filename, folder_name, output_path):
    """Download a file from Google Drive using the Drive API via google-auth."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import google.auth
        import io

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        service = build("drive", "v3", credentials=creds)

        # Find the file
        query = f"name = '{filename}'"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])

        if not files:
            print(f"   ⚠️  File '{filename}' tidak ditemukan di Drive.")
            print(f"   File mungkin masih diproses. Coba download manual dari Drive.")
            return False

        file_id = files[0]["id"]
        request = service.files().get_media(fileId=file_id)
        with open(output_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                dl_status, done = downloader.next_chunk()
                if dl_status:
                    pct = int(dl_status.progress() * 100)
                    print(f"      Download: {pct}%", end="\r")
            print()  # newline after progress
        return True

    except ImportError:
        print("   ⚠️  google-api-python-client belum terinstall untuk auto-download.")
        print("   Install dengan: pip install google-api-python-client")
        return False
    except Exception as e:
        print(f"   ⚠️  Drive API error: {e}")
        return False


def main():
    print("=" * 60)
    print("  Sentinel-2 Data Downloader — DKI Jakarta")
    print("=" * 60)
    print(f"  ROI (bbox): {BBOX}")
    print(f"  Resolusi  : {SCALE} m/px")
    print(f"  Band      : {', '.join(OUTPUT_BAND_NAMES)}")
    print("=" * 60)

    ee = initialize_ee()

    # Define region of interest
    roi = ee.Geometry.Rectangle(BBOX)

    success_count = 0
    for year, cfg in YEARS.items():
        print(f"\n{'─' * 50}")
        print(f"📅 Tahun {year}: {cfg['start']} → {cfg['end']}")
        print(f"{'─' * 50}")

        # Check if file already exists
        if os.path.exists(cfg["output"]):
            size_mb = os.path.getsize(cfg["output"]) / (1024 * 1024)
            print(f"   ⏭️  File sudah ada ({size_mb:.1f} MB), dilewati.")
            print(f"   Hapus file untuk download ulang: {cfg['output']}")
            success_count += 1
            continue

        # Build composite
        composite = get_s2_composite(ee, cfg["start"], cfg["end"], roi)
        if composite is None:
            continue

        # Download
        if download_geotiff(ee, composite, roi, cfg["output"], SCALE):
            success_count += 1

    print(f"\n{'=' * 60}")
    if success_count == len(YEARS):
        print("🎉 Semua data berhasil didownload!")
        print()
        print("Langkah selanjutnya:")
        print("  python prepare_patches_improved.py")
    else:
        print(f"⚠️  {success_count}/{len(YEARS)} file berhasil.")
        print("Periksa error di atas dan coba lagi.")
    print("=" * 60)


if __name__ == "__main__":
    main()
