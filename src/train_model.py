# ===================================================================
# 2. TRAINING LOGIC
# ===================================================================

import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross-Entropy loss for imbalanced segmentation."""
    def __init__(self, weight=0.5, smooth=1e-5):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to model outputs to get probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate Dice loss
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(inputs_flat, targets_flat, reduction='mean')

        # Combine losses
        return self.weight * bce_loss + (1 - self.weight) * dice_loss

def train_model(model, train_loader, valid_loader, device, models_dir, results_dir, epochs=25):
    """
    Main training function.

    Args:
        model: The neural network model.
        train_loader: DataLoader for the training set.
        valid_loader: DataLoader for the validation set.
        device: The device to train on ('cuda' or 'cpu').
        models_dir: Directory to save model checkpoints.
        results_dir: Directory to save plots.
        epochs (int): Number of epochs to train.
    """
    print("🚀 Starting training...")

    # --- Loss Function, Optimizer, Hyperparameters ---
    criterion = DiceBCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    history = {'train_loss': [], 'valid_loss': []}
    start_epoch = 0
    best_valid_loss = float('inf')


    # Find the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(models_dir, 'checkpoint_epoch_*.pth'))
    if checkpoint_files:
        latest_checkpoint_path = max(checkpoint_files, key=os.path.getctime)
        print(f"🔄 Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        best_valid_loss = checkpoint.get('best_valid_loss', float('inf')) # Use .get for backward compatibility
    else:
        print("🚀 Starting training from scratch.")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_train_loss = 0.0

        # Training loop
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation loop
        model.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_valid_loss += loss.item() * images.size(0)

        epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)
        history['valid_loss'].append(epoch_valid_loss)

        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.4f}, Valid Loss: {epoch_valid_loss:.4f}")


        # --- Save Checkpoint After Every Epoch ---
        is_best = epoch_valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = epoch_valid_loss
            print(f"✅ New best validation loss: {best_valid_loss:.4f}")

        checkpoint_path = os.path.join(models_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_valid_loss': best_valid_loss
        }, checkpoint_path)

        # Save a separate 'best_model.pth' for easy evaluation later
        if is_best:
            best_model_path = os.path.join(models_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Best model saved to {best_model_path}")

        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['valid_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(results_dir, 'training_loss_plot.png')
        plt.savefig(plot_path)
        print(f"📈 Loss plot saved to {plot_path}")

    print("🏁 Training complete.")