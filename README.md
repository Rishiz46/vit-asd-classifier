# 🧠 Vision Transformer for ASD Classification (ViT-ASD-Classifier)

This repository provides a complete pipeline to fine-tune a Vision Transformer (ViT) on functional MRI (fMRI) data to classify Autism Spectrum Disorder (ASD). It also includes code to extract embeddings for downstream multimodal fusion with TabTransformer and CMCL.

```### 📁 Repository Structure
vit-asd-classifier/
├── vit_model/ # Fine-tuned ViT model weights (.pth) and zipped model (.zip)
├── embeddings/ # Extracted embeddings (for CMCL)
│ ├── vit_embeddings.npy
│ └── vit_labels.npy
├── dataset/ # Input data and CSVs
│ └── Phenotypic_V1_0b_preprocessed1.csv

     # use the dataset_download.py & unzip_preprocess.py
     to download the dataset & pre-processing
     (I do dataset & pre-processing on my local, after I fine-tune the VIT model in Kaggle because it's faster)

├── scripts/ # Python code files
│ ├── Dataset & CSV downloading script [Local Version].py # Download and zip the dataset
│ ├── Unzip & Preprocessing Script [Local Version] .py # Unzip and preprocess dataset
│ ├── ViT Fine-Tune [Kaggle Version] Script.py # Train ViT model
│ ├── Plot_Metrics [Kaggle Version].py
│ ├── Zip_Model [Kaggle Version].py
│ └── Extract-Embedding For Fusion [Kaggle version].py # Extract embeddings for CMCL
├── requirements.txt # All Python package dependencies
└── README.md # This file

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Rishiz46/vit-asd-classifier.git
cd vit-asd-classifier

### 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate' '' On Windows: venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

📦 How to Run

✅ Step 1: Download and preprocess the dataset

python scripts/Dataset & CSV downloading script [Local Version].py
python scripts/Unzip & Preprocessing Script [Local Version] .py
This will generate .npy files used for training.

✅ Step 2: Train the Vision Transformer model

python scripts/ViT Fine-Tune [Kaggle Version] Script.py
Fine-tunes ViT (google/vit-base-patch16-224-in21k)
Saves the best model to vit_model/best_model.pth
Uses Focal Loss, augmentation, and early stopping

✅ Step 3: Extract fMRI embeddings for fusion

python scripts/Extract-Embedding For Fusion [Kaggle version].py
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

