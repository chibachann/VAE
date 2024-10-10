import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from torchvision.utils import make_grid, save_image
import os

from model.model import VAE
from config.config import Config

config = Config()
config.latent_dim = 10

input_dim = config.input_dim
hidden_dim = config.hidden_dim
latent_dim = config.latent_dim


# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルのロード
output_dir = f'output_latent_{latent_dim}'
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
model.load_state_dict(torch.load(f'{output_dir}/vae_latent_{latent_dim}.pth', map_location=device))
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
centers = kmeans.cluster_centers_


# クラスター中心間の補間と画像生成
def interpolate_centers(model, start, end, device, steps=10):
    model.eval()
    with torch.no_grad():
        interpolated_images = []
        for t in np.linspace(0, 1, steps):
            z = torch.from_numpy((1-t) * start + t * end).float().unsqueeze(0).to(device)
            generated = model.decode(z)
            interpolated_images.append(generated.cpu())
        
        interpolated_images = torch.cat(interpolated_images, dim=0)
        return interpolated_images.view(-1, 1, 28, 28)

# 出力ディレクトリの作成
output_dir = f'{output_dir}/center_interpolations'
os.makedirs(output_dir, exist_ok=True)

# クラスター中心間の補間と画像生成
for i in range(len(centers)):
    for j in range(i+1, len(centers)):
        start = centers[i]
        end = centers[j]
        
        interpolated_images = interpolate_centers(model, start, end, device)
        
        # 補間画像の保存
        save_image(interpolated_images, f'{output_dir}/interpolation_{i}_to_{j}.png', nrow=10)
        
        # 潜在空間の可視化（補間経路を含む）
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap='tab10', alpha=0.5)
        plt.colorbar(scatter)
        
        # クラスター中心のプロット
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        
        # 補間経路の描画
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2, alpha=0.8)
        plt.scatter([start[0], end[0]], [start[1], end[1]], c='red', s=100, zorder=5)
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title(f'Latent Space Interpolation: Center {i} to Center {j}')
        plt.savefig(f'{output_dir}/latent_space_{i}_to_{j}.png')
        plt.close()

print(f"Interpolations saved in '{output_dir}'")

# 全体の潜在空間の可視化
plt.figure(figsize=(12, 10))
scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap='tab10', alpha=0.5)
plt.colorbar(scatter)

# クラスター中心のプロット
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

# すべての補間経路の描画
for i in range(len(centers)):
    for j in range(i+1, len(centers)):
        start = centers[i]
        end = centers[j]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1, alpha=0.3)

plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space Visualization with All Interpolation Paths')
plt.savefig(f'{output_dir}/latent_space_all_paths.png')
plt.close()

print(f"Overall latent space visualization saved as '{output_dir}/latent_space_all_paths.png'")