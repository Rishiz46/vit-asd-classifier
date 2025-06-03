# 🧠 Vision Transformer for ASD Classification (ViT-ASD-Classifier)

This repository provides a complete pipeline to fine-tune a Vision Transformer (ViT) on functional MRI (fMRI) data to classify Autism Spectrum Disorder (ASD). It also includes code to extract embeddings for downstream multimodal fusion with TabTransformer and CMCL.

## 📁 Repository Structure

vit-asd-classifier/
│

├── vit_model/ # Fine-tuned model weights (.pth) and zipped model (.zip)

├── embeddings/ # Extracted embeddings (for CMCL)

│ ├── vit_embeddings.npy

│ ├── vit_labels.npy
│
├── dataset/ # Input data and CSVs

│ ├── raw/ # Original zipped files (optional)

│ ├── processed/ # Preprocessed .npy files

│ └── Phenotypic_V1_0b.csv
│
├── scripts/ # Python code files

│ ├── dataset_download.py # Download and zip dataset

│ ├── unzip_preprocess.py # Unzip and preprocess dataset

│ ├── train_vit.py # Train ViT model

│ └── extract_embeddings.py # Extract embeddings for CMCL
│

├── requirements.txt # All Python package dependencies

└── README.md # This file

---
## 🚀 Quick Start

### 1. Clone the repository
`bash
git clone https://github.com/Rishiz46/vit-asd-classifier.git
cd vit-asd-classifier

### 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  `` On Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

📦 How to Run

✅ Step 1: Download and preprocess dataset

python scripts/dataset_download.py
python scripts/unzip_preprocess.py
This will generate .npy files used for training.

✅ Step 2: Train the Vision Transformer model

python scripts/train_vit.py
Fine-tunes ViT (google/vit-base-patch16-224-in21k)
Saves the best model to vit_model/best_model.pth
Uses Focal Loss, augmentation, early stopping

✅ Step 3: Extract fMRI embeddings for fusion

python scripts/extract_embeddings.py
Saves vit_embeddings.npy and vit_labels.npy to the embeddings/ directory
These are used later in CMCL for multimodal fusion

📊 Outputs
✅ Training Logs: Accuracy, Loss, Classification Report
✅ Visualizations: Accuracy Curve, ROC, PCA, t-SNE
✅ Extracted Features for CMCL

📚 Dataset & Model Sources
fMRI Data: ABIDE
Pretrained ViT: google/vit-base-patch16-224-in21k

✅ Future Work
 Train TabTransformer on phenotypic data
 Apply CMCL for fusion of vision + tabular features
 Build a user-facing web app for inference

🤝 Acknowledgements
Hugging Face Transformers
ABIDE fMRI Dataset
CMCL Framework for fusion

📬 Contact
If you use this project or have any questions, feel free to:

🌐 Raise an issue
💬 Submit a pull request
📩 Email: rishikesavan500@gmail.com
