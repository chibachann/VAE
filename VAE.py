import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torch.optim as optim

from model.model import VAE
from config.config import Config
from src.function import train

# Instantiate configuration
config = Config()
config.latent_dim = 10

# Use configuration parameters
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate
input_dim = config.input_dim
hidden_dim = config.hidden_dim
latent_dim = config.latent_dim

# Create output directory based on latent_dim
output_dir = f'output_latent_{latent_dim}'
os.makedirs(output_dir, exist_ok=True)



# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and optimizer
model = VAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training the VAE model with latent_dim = {latent_dim}...")


train(model, train_loader, optimizer, device, epochs, output_dir)


# Save the model
model_save_path = f'{output_dir}/vae_latent_{latent_dim}.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")