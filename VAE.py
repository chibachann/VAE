import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os
from torchvision.utils import save_image

import torch.nn as nn
import torch.optim as optim

# Configuration class
class Config:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 10
        self.learning_rate = 1e-3
        self.input_dim = 784
        self.hidden_dim = 400
        self.latent_dim = 20

# Instantiate configuration
config = Config()

# Use configuration parameters
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate
input_dim = config.input_dim
hidden_dim = config.hidden_dim
latent_dim = config.latent_dim

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Data loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model, optimizer and writer
model = VAE(input_dim, hidden_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter('output/tensorboard')

print("Training the VAE model...")

"""
# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data)}')

    print(f'====> Epoch {epoch} Average loss: {train_loss / len(train_loader.dataset)}')
    writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)

    # Save reconstructed images
    with torch.no_grad():
        sample = torch.randn(64, latent_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 1, 28, 28), f'output/reconstruction_epoch_{epoch}.png')
        

# Save the model
torch.save(model.state_dict(), 'output/vae.pth')
writer.close()

# Function to save model checkpoints
def save_checkpoint(model, optimizer, epoch, path='output/checkpoint.pth'):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)

# Function to load model checkpoints
def load_checkpoint(model, optimizer, path='output/checkpoint.pth'):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']
    else:
        return 0

# Save the final model checkpoint
save_checkpoint(model, optimizer, epochs)
"""
