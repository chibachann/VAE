import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

from model.model import VAE
from config.config import Config

config = Config()

input_dim = config.input_dim
hidden_dim = config.hidden_dim
latent_dim = config.latent_dim


# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルのロード
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
model.load_state_dict(torch.load('output/vae.pth', map_location=device))
model.eval()

# データセットの準備
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# 潜在空間の表現を取得
z_points = []
labels = []

with torch.no_grad():
    for data, label in dataloader:
        data = data.to(device)
        mu, _ = model.encode(data.view(-1, 784))
        z_points.append(mu.cpu().numpy())
        labels.append(label.numpy())

z_points = np.concatenate(z_points, axis=0)
labels = np.concatenate(labels, axis=0)

# KMeansでクラスタリング
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(z_points)

# 可視化
plt.figure(figsize=(12, 10))
scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap='tab10', alpha=0.5)
plt.colorbar(scatter)

# クラスターの中心を追加
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space Visualization with Cluster Centers')
plt.savefig('output/latent_space_clusters.png')
plt.close()

print("Visualization saved as 'output/latent_space_clusters.png'")