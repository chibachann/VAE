import torch
from torchvision.utils import save_image

import torch.nn as nn

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Function to save original and reconstructed images side by side
def save_comparison(original, reconstructed, epoch, output_dir):
    # Ensure both tensors have the same shape
    original = original.view(-1, 1, 28, 28)
    reconstructed = reconstructed.view(-1, 1, 28, 28)
    comparison = torch.cat([original[:8], reconstructed[:8]])
    save_image(comparison.cpu(),
               f'{output_dir}/comparison_epoch_{epoch}.png',
               nrow=8)

def train(model, train_loader, optimizer, device, epochs, output_dir):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.4f}')

        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch {epoch} Average loss: {avg_loss:.4f}')

        # Save original and reconstructed images
        with torch.no_grad():
            # Get a batch of test data
            test_data = next(iter(train_loader))[0].to(device)
            # Reconstruct the test data
            recon_data, _, _ = model(test_data)
            # Ensure the reconstructed data has the correct shape
            recon_data = recon_data.view(-1, 1, 28, 28)
            # Save the comparison
            save_comparison(test_data, recon_data, epoch, output_dir)