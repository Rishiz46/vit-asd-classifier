import os
import requests
import zipfile
import shutil
from nilearn.datasets import fetch_abide_pcp
import pandas as pd

# === Setup Paths ===
target_dir = r"C:\Users\rishi\OneDrive\Documents\Final year project [Batch 20]\dataset"
data_dir = os.path.join(target_dir, "abide_subset")
csv_filename = "Phenotypic_V1_0b_preprocessed1.csv"
zip_filename = os.path.join(target_dir, "abide_subset.zip")

# === Step 1: Download Phenotypic CSV ===
csv_url = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE/Phenotypic_V1_0b_preprocessed1.csv'
csv_path = os.path.join(data_dir, csv_filename)

os.makedirs(data_dir, exist_ok=True)
print("📥 Downloading phenotypic CSV...")

try:
    response = requests.get(csv_url)
    response.raise_for_status()
    with open(csv_path, 'wb') as f:
        f.write(response.content)
    print("✅ CSV downloaded:", csv_path)
except Exception as e:
    print("❌ CSV download failed:", e)

# === Step 2: Download ABIDE fMRI Subset ===
print("📥 Downloading fMRI data subset...")
abide = fetch_abide_pcp(
    n_subjects=700,
    pipeline='cpac',
    band_pass_filtering=True,
    global_signal_regression=True,
    data_dir=data_dir
)

print("✅ fMRI files downloaded to:", data_dir)

# === Step 3: Zip the folder ===
print("🗜️ Zipping dataset...")
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(data_dir):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, start=target_dir)
            zipf.write(filepath, arcname)

print("✅ Zip file created:", zip_filename)

# === Optional: Cleanup extracted folder ===
shutil.rmtree(data_dir)
print("🧹 Cleaned up original dataset folder.")
