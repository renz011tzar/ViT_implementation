import torch
from pathlib import Path
import os

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    checkpoint_dir: str = "modelos"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file_path = checkpoint_dir_path / f"checkpoint_epoch_{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_file_path)

def load_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device):
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}")
        return model, optimizer, start_epoch
    else:
        print("No checkpoint found at specified path.")
        return model, optimizer, 0
