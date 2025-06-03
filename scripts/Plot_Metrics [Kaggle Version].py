import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Simulated inputs (replace with actual model output in real use)
val_accs = [0.55, 0.56, 0.57, 0.59, 0.58, 0.60, 0.61, 0.62, 0.60, 0.59]
labels = np.random.randint(0, 2, size=138)
probs = np.random.rand(138)
probs = np.stack([1 - probs, probs], axis=1)
features = np.random.randn(138, 768)  # <-- Replace with model CLS token features

# =====================
# 1. Normalize Features
# =====================
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# ===========================
# 2. Validation Accuracy Plot
# ===========================
plt.plot(val_accs, marker='o', label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()

# =====================
# 3. ROC Curve + AUC
# =====================
fpr, tpr, _ = roc_curve(labels, probs[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# =====================
# 4. PCA Visualization
# =====================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_std)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
plt.title("PCA of ViT Features")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar()
plt.show()

# ========================
# 5. t-SNE Visualization
# ========================
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
tsne_result = tsne.fit_transform(features_std)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
plt.title("t-SNE of ViT Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar()
plt.show()
