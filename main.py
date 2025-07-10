# main_trainer.py
import torch
from torchvision import transforms
import torch.optim as optim
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import multiprocessing as mp
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import gc # Kept for explicit calls if needed, but loop usage is removed.

# import voxelmorph2d 
import voxelmorph2d as vm2d

# ---Define device once ---
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.cuda.manual_seed(42)


class VoxelMorphTrainer:
    """
    A modernized trainer class for the VoxelMorph2D model.
    It encapsulates the training loop, validation, checkpointing, and visualization.
    """
    def __init__(self, input_dims, in_channels):
        self.input_dims = input_dims # e.g., (256, 256)
        
        # The model is moved to the correct device upon initialization.
        self.model = vm2d.VoxelMorph2d(in_channels=in_channels).to(device)
        
        # Optimizer and Scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)

        # Checkpoint and history tracking
        self.checkpoint_dir = './checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history = {
            'train_loss': [], 'train_dice': [],
            'val_loss': [], 'val_dice': []
        }
        self.best_val_dice = 0.0

    def train_step(self, moving_batch, fixed_batch, alpha, lambda_reg):
        """Performs a single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        moving_batch = moving_batch.to(device)
        fixed_batch = fixed_batch.to(device)

        registered_image, deform_field = self.model(moving_batch, fixed_batch)
        
        loss = vm2d.combined_loss(
            registered_image, fixed_batch, deform_field,n=25,
            alpha=alpha, lambda_reg=lambda_reg
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Gradient clipping
        self.optimizer.step()
        
        with torch.no_grad():
            dice_score = 1 - vm2d.dice_loss(registered_image, fixed_batch)
        
        return loss.item(), dice_score.item()

    def validation_step(self, moving_batch, fixed_batch, alpha, lambda_reg):
        """Performs a single validation step."""
        self.model.eval()
        with torch.no_grad():
            moving_batch = moving_batch.to(device)
            fixed_batch = fixed_batch.to(device)

            registered_image, deform_field = self.model(moving_batch, fixed_batch)
            
            loss = vm2d.combined_loss(
                registered_image, fixed_batch, deform_field,n=25,
                alpha=alpha, lambda_reg=lambda_reg
            )
            dice_score = 1 - vm2d.dice_loss(registered_image, fixed_batch)
        
        return loss.item(), dice_score.item()

    def save_checkpoint(self, epoch, is_best=False):
        """Saves model checkpoint."""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_dice': self.best_val_dice
        }
        filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, filename)
        if is_best:
            best_filename = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_filename)
            print(f"** Best model saved to {best_filename} **")

    def load_checkpoint(self, filename):
        """Loads model checkpoint."""
        if not os.path.isfile(filename):
            print(f"=> No checkpoint found at '{filename}'")
            return 0
        
        print(f"=> Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.history = checkpoint.get('history', self.history)
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        start_epoch = checkpoint['epoch']
        print(f"=> Loaded checkpoint '{filename}' (epoch {start_epoch})")
        return start_epoch

# --- Data Augmentation ---
IMG_SIZE = (256, 256)
train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Uncomment if images are not [0,1]
])

val_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Must match train transforms
])

class PairedImageDataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, file_ids, data_dir, transform=None):
        self.file_ids = file_ids
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = self.file_ids[index]
        
        # Load moving and fixed images
        # Ensure your filenames follow a consistent pattern
        moving_path = os.path.join(self.data_dir, f'{file_id}_2.png')
        fixed_path = os.path.join(self.data_dir, f'{file_id}_1.png')
        
        # This handles grayscale PNGs gracefully.
        moving_img = Image.open(moving_path).convert('RGB')
        fixed_img = Image.open(fixed_path).convert('RGB')
        
        if self.transform:
            # Apply the same seed to transforms for both images if spatial transforms should match
            # For independent augmentation, this is fine.
            moving_img = self.transform(moving_img)
            fixed_img = self.transform(fixed_img)
        
        # ToTensor() already provides the correct [C, H, W] format.
        return moving_img, fixed_img


def visualize_history(history, save_path='training_history.png'):
    """Visualizes training and validation history."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(history['train_loss'], label='Train Loss', color='royalblue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='darkorange', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training & Validation Loss')
    
    ax2.plot(history['train_dice'], label='Train Dice', color='royalblue')
    ax2.plot(history['val_dice'], label='Validation Dice', color='darkorange', linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.set_title('Training & Validation Dice Score')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # --- Configuration ---
    DATA_DIR = './images/'
    NUM_CHANNELS = 3 # For RGB images
    PARAMS = {
        'batch_size': 8, # Adjust based on your GPU memory
        'shuffle': True,
        'num_workers': min(mp.cpu_count(), 4),
        'pin_memory': device.type == 'cuda'
    }
    MAX_EPOCHS = 500
    EARLY_STOPPING_PATIENCE = 50

    # --- Initialize Trainer ---
    # The model expects the concatenated channel count of moving and fixed images.
    trainer = VoxelMorphTrainer(input_dims=IMG_SIZE, in_channels=NUM_CHANNELS * 2)

    # --- Data Loading ---
    all_file_ids = list(set([f.split('_')[0] for f in os.listdir(DATA_DIR) if f.endswith('.png')]))
    train_ids, val_ids = train_test_split(all_file_ids, test_size=0.2, random_state=42)

    train_set = PairedImageDataset(train_ids, DATA_DIR, transform=train_transforms)
    train_loader = data.DataLoader(train_set, **PARAMS)
    
    val_set = PairedImageDataset(val_ids, DATA_DIR, transform=val_transforms)
    val_loader = data.DataLoader(val_set, **PARAMS)

    # --- Training Loop ---
    start_epoch = trainer.load_checkpoint(os.path.join(trainer.checkpoint_dir, 'model_best.pth'))
    no_improvement_count = 0

    print(f"🚀 Starting training from epoch {start_epoch + 1} on {device}")
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start_time = time.time()
        
        # Dynamic loss weighting
        alpha = min(0.3 + 0.4 * (epoch / 200), 0.7) # Gradually increase Dice weight
        lambda_reg = 0.01

        # Training phase
        train_loss_epoch, train_dice_epoch = 0, 0
        for moving_batch, fixed_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]"):
            loss, dice = trainer.train_step(moving_batch, fixed_batch, alpha, lambda_reg)
            train_loss_epoch += loss
            train_dice_epoch += dice
        
        trainer.history['train_loss'].append(train_loss_epoch / len(train_loader))
        trainer.history['train_dice'].append(train_dice_epoch / len(train_loader))
        
        # Validation phase
        val_loss_epoch, val_dice_epoch = 0, 0
        for moving_batch, fixed_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]"):
            loss, dice = trainer.validation_step(moving_batch, fixed_batch, alpha, lambda_reg)
            val_loss_epoch += loss
            val_dice_epoch += dice
            
        trainer.history['val_loss'].append(val_loss_epoch / len(val_loader))
        trainer.history['val_dice'].append(val_dice_epoch / len(val_loader))

        trainer.scheduler.step() # Update learning rate

        # Logging
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} ({epoch_time:.2f}s) | "
              f"Train Loss: {trainer.history['train_loss'][-1]:.4f}, Train Dice: {trainer.history['train_dice'][-1]:.4f} | "
              f"Val Loss: {trainer.history['val_loss'][-1]:.4f}, Val Dice: {trainer.history['val_dice'][-1]:.4f}")

        # Checkpoint and Early Stopping
        current_val_dice = trainer.history['val_dice'][-1]
        if current_val_dice > trainer.best_val_dice:
            trainer.best_val_dice = current_val_dice
            trainer.save_checkpoint(epoch + 1, is_best=True)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(epoch + 1)
        
        # Visualize history
        visualize_history(trainer.history)
        
        if no_improvement_count >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping after {EARLY_STOPPING_PATIENCE} epochs with no improvement.")
            break

    print(f"🏁 Training finished. Best validation Dice score: {trainer.best_val_dice:.4f}")

if __name__ == "__main__":
    main()
