import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor
from tqdm import tqdm
import numpy as np

# Load fine-tuned model (same weights as your classifier model)
vit_model = AutoModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.load_state_dict(torch.load("/kaggle/working/vit_model/best_model.pth"), strict=False)
vit_model.to("cuda")
vit_model.eval()

# DataLoader for val or full dataset
vit_features = []
vit_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader):  # or train_loader
        inputs = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
        labels = batch["labels"]
        
        # Extract last hidden state of [CLS] token
        outputs = vit_model(**inputs).last_hidden_state  # shape: (batch_size, 197, 768)
        cls_embeddings = outputs[:, 0, :]  # only [CLS] token

        vit_features.append(cls_embeddings.cpu().numpy())
        vit_labels.append(labels.numpy())

# Save features
vit_features = np.concatenate(vit_features, axis=0)
vit_labels = np.concatenate(vit_labels, axis=0)
np.save("vit_embeddings.npy", vit_features)
np.save("vit_labels.npy", vit_labels)
