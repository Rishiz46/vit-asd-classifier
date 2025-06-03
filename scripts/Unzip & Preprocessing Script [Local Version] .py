import os
import zipfile
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

# === CONFIG ===
main_zip_path = r"C:\Users\rishi\OneDrive\Documents\Final year project [Batch 20]\dataset\abide_subset.zip"
extract_main_to = r"C:\Users\rishi\OneDrive\Documents\Final year project [Batch 20]\dataset\abide_unzipped"
preprocessed_output = r"C:\Users\rishi\OneDrive\Documents\Final year project [Batch 20]\dataset\abide_unzipped\preprocessed"

# === Ensure directories exist ===
os.makedirs(extract_main_to, exist_ok=True)
os.makedirs(preprocessed_output, exist_ok=True)

# === Step 1: Unzip the main ZIP file ===
print("🔓 Extracting main ZIP file...")
with zipfile.ZipFile(main_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_main_to)
print("✅ Main ZIP extracted.")

# === Step 2: Search for all .nii.gz files ===
print("🔍 Searching for all .nii.gz files (including nested)...")
nii_files = []
for root, _, files in os.walk(extract_main_to):
    for file in files:
        if file.endswith('.nii.gz'):
            nii_files.append(os.path.join(root, file))

print(f"🧠 Found {len(nii_files)} fMRI files to process.")

# === Preprocessing Function ===
def preprocess_fmri(filepath, output_path, target_size=(128, 128)):
    try:
        img = nib.load(filepath)
        data = img.get_fdata()

        t = np.random.randint(data.shape[-1])
        volume = data[:, :, :, t]

        slices = []
        for offset in [-1, 0, 1]:
            z = volume.shape[2] // 2 + offset
            if 0 <= z < volume.shape[2]:
                slc = volume[:, :, z]
                slc = (slc - np.min(slc)) / (np.max(slc) - np.min(slc) + 1e-8)
                slices.append(slc)

        if len(slices) < 3:
            print(f"❌ Skipping {filepath}: not enough slices")
            return

        stacked = np.stack(slices, axis=0)
        img_tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)
        resized = F.interpolate(img_tensor, size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        np.save(output_path, resized.numpy())
    except Exception as e:
        print(f"⚠️ Failed to process {filepath}: {e}")

# === Apply Preprocessing ===
for file_path in tqdm(nii_files, desc="Preprocessing"):
    filename = os.path.basename(file_path).replace(".nii.gz", ".npy")
    output_path = os.path.join(preprocessed_output, filename)
    preprocess_fmri(file_path, output_path)

print(f"\n✅ Done. Preprocessed files saved to:\n{preprocessed_output}")
