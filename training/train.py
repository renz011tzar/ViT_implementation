import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from DP_implementation.util.checkpoint import *
import os

def train_model_1(net, criterion, optimizer, tloader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    #every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def train_test_1(model: torch.nn.Module, 
                       dataloader: torch.utils.data.DataLoader, 
                       loss_fn: torch.nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       device: torch.device, 
                       train: bool) -> Tuple[float, float]:
    model.train() if train else model.eval()
    running_loss, correct_preds, total_preds = 0.0, 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        running_loss += loss.item() * X.size(0)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        _, predicted = torch.max(y_pred, 1)
        correct_preds += (predicted == y).sum().item()
        total_preds += y.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          device: torch.device,
          checkpoint_dir: str = "models") -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # Load the number of epochs from the environment variable
    epochs = int(os.getenv("EPOCH_NUM", 10))  # Default to 10 if not set

    # Load checkpoint if USE_CHECKPOINT is set to True
    if os.getenv("USE_CHECKPOINT", "False") == "True":
        checkpoint_path = os.getenv("CHECKPOINT_PATH", "")
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint from {checkpoint_path} at epoch {start_epoch}")
        else:
            print("No checkpoint path provided. Training from scratch.")
            start_epoch = 0
    else:
        print("Not using checkpoint. Training from scratch.")
        start_epoch = 0

    for epoch in tqdm(range(start_epoch, epochs)):
        train_loss, train_acc = train_test_1(model, train_dataloader, loss_fn, optimizer, device, train=True)
        test_loss, test_acc = train_test_1(model, test_dataloader, loss_fn, optimizer, device, train=False)

        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        save_checkpoint(model=model,
                        optimizer=optimizer,
                        epoch=epoch+1,
                        target_dir=checkpoint_dir,
                        checkpoint_name=f"checkpoint_epoch_{epoch+1}.pth")

    return results

