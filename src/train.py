import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from dataset import get_dataloaders, EMOTIONS
from model import build_model, get_device

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = '../data'
SAVE_DIR    = '../models'
BATCH_SIZE  = 64
NUM_CLASSES = 7

PHASE1_EPOCHS = 5
PHASE1_LR     = 0.001
PHASE2_EPOCHS = 30
PHASE2_LR     = 0.0001
PATIENCE      = 7


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted   = outputs.argmax(dim=1)
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader), 100 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs  = model(images)
            loss     = criterion(outputs, labels)
            total_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)
    return total_loss / len(loader), 100 * correct / total


def plot_curves(history, phase, save_dir):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training curves - Phase {phase}", fontsize=14)

    ax1.plot(epochs, history["train_acc"], "b-o", label="Train")
    ax1.plot(epochs, history["val_acc"],   "r-o", label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_loss"], "b-o", label="Train")
    ax2.plot(epochs, history["val_loss"],   "r-o", label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f"curves_phase{phase}.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved training curves to {path}")


def train(phase, model, train_loader, val_loader, epochs, lr, device, save_dir):
    print(f"\n{'='*50}")
    print(f"  PHASE {phase} TRAINING")
    print(f"  Epochs: {epochs}  |  LR: {lr}")
    print(f"{'='*50}\n")

    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc     = 0.0
    patience_counter = 0
    best_model_path  = os.path.join(save_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        print(f"Epoch [{epoch:02d}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "phase": phase,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, best_model_path)
            print(f"  --> Best model saved (val_acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs.")
                break

    plot_curves(history, phase, save_dir)
    print(f"\nPhase {phase} complete. Best val accuracy: {best_val_acc:.2f}%")
    return model, best_val_acc


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = get_device()

    print("Loading data...")
    train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

    print("Building model...")
    model = build_model(pretrained=True, freeze_backbone=True)
    model = model.to(device)

    model, phase1_acc = train(1, model, train_loader, val_loader,
                              PHASE1_EPOCHS, PHASE1_LR, device, SAVE_DIR)

    print("\nUnfreezing backbone for fine tuning...")
    for param in model.features.parameters():
        param.requires_grad = True

    model, phase2_acc = train(2, model, train_loader, val_loader,
                              PHASE2_EPOCHS, PHASE2_LR, device, SAVE_DIR)

    print(f"\nTRAINING COMPLETE")
    print(f"Phase 1 best accuracy : {phase1_acc:.2f}%")
    print(f"Phase 2 best accuracy : {phase2_acc:.2f}%")
    print(f"Best model saved at   : {SAVE_DIR}/best_model.pth")