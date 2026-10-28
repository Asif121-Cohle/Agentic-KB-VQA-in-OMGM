# ==============================================================================
# Google Colab Script: InfoSeek Train Verifier Query Images (3,300 Images)
# ==============================================================================
# Workflow:
#   1. Mounts Google Drive
#   2. Downloads master archive (BByrneLab/M2KR_Images -> Infoseek/infoseek_train_images.tar)
#   3. Selectively extracts ONLY the ~3,292 required train/val images from required_oven_images_3300.txt
#   4. Compresses them into 'infoseek_train_3300_images.zip' (~160 MB)
#   5. Saves directly to Google Drive for upload to Kaggle
#
# Prerequisite:
#   Upload 'required_oven_images_3300.txt' to Colab root (/content/)
# ==============================================================================

# --- [CELL 1]: Mount Google Drive & Set Paths ---
from google.colab import drive
import os
import shutil

print("Mounting Google Drive...")
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/OMGM_Data"
os.makedirs(DRIVE_DIR, exist_ok=True)

LOCAL_TAR = "/content/infoseek_train_images.tar"
DRIVE_TAR = os.path.join(DRIVE_DIR, "infoseek_train_images.tar")
REQUIRED_IDS_FILE = "/content/required_oven_images_3300.txt"
OUTPUT_DIR = "/content/infoseek_train_3300_images"
LOCAL_ZIP = "/content/infoseek_train_3300_images.zip"
DRIVE_ZIP = os.path.join(DRIVE_DIR, "infoseek_train_3300_images.zip")

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Drive output location: {DRIVE_DIR}")
print(f"Extraction directory:  {OUTPUT_DIR}")


# --- [CELL 2]: Install Dependencies & Download Train Images Tar ---
!pip install -q huggingface_hub tqdm

from huggingface_hub import hf_hub_download

if os.path.exists(DRIVE_TAR):
    print(f"Found master archive in Google Drive: {DRIVE_TAR}")
    print("Copying to local Colab SSD...")
    shutil.copy2(DRIVE_TAR, LOCAL_TAR)
else:
    print("Downloading master infoseek_train_images.tar (44 GB) via Hugging Face CDN...")
    downloaded_path = hf_hub_download(
        repo_id="BByrneLab/M2KR_Images",
        filename="Infoseek/infoseek_train_images.tar",
        repo_type="dataset",
        local_dir="/content/"
    )
    if downloaded_path != LOCAL_TAR and os.path.exists(downloaded_path):
        shutil.move(downloaded_path, LOCAL_TAR)
    print("Download complete!")


# --- [CELL 3]: Match & Extract ONLY the 3,300 Images ---
import tarfile
from tqdm import tqdm

if not os.path.exists(REQUIRED_IDS_FILE):
    raise FileNotFoundError(f"Please upload '{REQUIRED_IDS_FILE}' to /content/ first!")

with open(REQUIRED_IDS_FILE, "r") as f:
    target_filenames = set(line.strip() for line in f if line.strip())

# Clean format
target_basenames = {os.path.basename(x).lower(): x for x in target_filenames}
target_stems = {os.path.splitext(os.path.basename(x))[0].lower(): x for x in target_filenames}

print(f"Loaded {len(target_filenames)} target image IDs to extract.")

found_ids = set()
print("\nExtracting matching images from master archive...")
with tarfile.open(LOCAL_TAR, "r:") as tar:
    for member in tqdm(tar, desc="Scanning & Extracting"):
        if not member.isfile():
            continue
        
        base_name = os.path.basename(member.name).lower()
        stem, _ = os.path.splitext(base_name)
        
        if (base_name in target_basenames) or (stem in target_stems):
            save_name = f"{stem}.jpg"
            dest_path = os.path.join(OUTPUT_DIR, save_name)
            source = tar.extractfile(member)
            if source:
                with open(dest_path, "wb") as out_f:
                    out_f.write(source.read())
                found_ids.add(stem)
                if len(found_ids) >= len(target_filenames):
                    print(f"\nAll {len(found_ids)} images found! Stopping extraction early.")
                    break

missing_ids = set(target_stems.keys()) - found_ids

print("\n" + "=" * 50)
print("EXTRACTION SUMMARY:")
print(f"  - Target requested: {len(target_filenames)}")
print(f"  - Found & extracted: {len(found_ids)}")
print(f"  - Missing:          {len(missing_ids)}")
print("=" * 50)


# --- [CELL 4]: Zip Extracted Images & Save to Drive ---
import zipfile

print(f"\nCompressing {len(found_ids)} clean images into zip...")
with zipfile.ZipFile(LOCAL_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in os.listdir(OUTPUT_DIR):
        zipf.write(os.path.join(OUTPUT_DIR, f), arcname=f)

zip_mb = os.path.getsize(LOCAL_ZIP) / (1024**2)
print(f"Zip created! Size: {zip_mb:.2f} MB")

print(f"Saving zip to Google Drive: {DRIVE_ZIP}...")
shutil.copy2(LOCAL_ZIP, DRIVE_ZIP)
print(f"SUCCESS! Clean dataset saved to Google Drive: {DRIVE_ZIP}")
