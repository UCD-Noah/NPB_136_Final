import torch
import umap
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

# 1. Load Pretrained CELL-DINO
# (Assuming the model is available via timm, HuggingFace, or a local PyTorch checkpoint)
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16') # Example generic DINO
model.eval()
model.cuda()

# 2. Extract Embeddings
def get_embeddings(dataloader, model):
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.cuda()
            # Extract the [CLS] token representation
            features = model(images) 
            embeddings.append(features.cpu().numpy())
            labels.append(targets.numpy())
            
    return np.vstack(embeddings), np.concatenate(labels)

# Assuming 'train_loader' is your dataset (Malaria or Microglia)
X_features, y_true_labels = get_embeddings(train_loader, model)

# 3. Apply UMAP
# n_neighbors and min_dist are crucial hyperparameters to tune for visual separation
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_features)

# 4. Clustering (K-Means)
# If testing Malaria, n_clusters=2 (Uninfected vs Parasitized)
# If testing Microglia, it could be 2 (Healthy vs Diseased) or more for intermediate states
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_umap) # or fit on X_features directly

# 5. Evaluate "Diagnosis" Capability
# Compare the unsupervised clusters to the ground truth labels
ari_score = adjusted_rand_score(y_true_labels, cluster_labels)
print(f"Clustering Accuracy (Adjusted Rand Index): {ari_score:.4f}")

# 6. Visualization
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap='viridis', s=5)
plt.title("UMAP Projection of CELL-DINO Embeddings")
plt.show()