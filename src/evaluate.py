import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders, EMOTIONS
from model import build_model, get_device


def load_trained_model(model_path, device):
    model = build_model(pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")
    print(f"Saved at epoch {checkpoint['epoch']} "
          f"with val_acc {checkpoint['val_acc']:.2f}%")
    return model


def get_all_predictions(model, loader, device):
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=EMOTIONS)
    disp.plot(ax=ax, colorbar=True, cmap='Blues')
    ax.set_title('Confusion Matrix — Facial Expression Model', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def print_classification_report(labels, preds, save_dir):
    report = classification_report(labels, preds,
                                   target_names=EMOTIONS,
                                   digits=3)
    print("\nClassification Report:")
    print("=" * 60)
    print(report)

    # Save to file
    path = os.path.join(save_dir, 'classification_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"Saved classification report to {path}")


def plot_per_emotion_accuracy(labels, preds, save_dir):
    cm = confusion_matrix(labels, preds)
    per_emotion_acc = cm.diagonal() / cm.sum(axis=1) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(EMOTIONS, per_emotion_acc, color='steelblue', edgecolor='black')
    ax.set_title('Per-Emotion Accuracy', fontsize=14)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.axhline(y=59.03, color='red', linestyle='--', label='Overall accuracy 59.03%')
    ax.legend()

    for bar, acc in zip(bars, per_emotion_acc):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, 'per_emotion_accuracy.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved per-emotion accuracy to {path}")


if __name__ == '__main__':
    DATA_DIR  = '../data'
    MODEL_PATH = '../models/best_model.pth'
    SAVE_DIR  = '../models'

    device = get_device()

    print("Loading model...")
    model = load_trained_model(MODEL_PATH, device)

    print("\nLoading test data...")
    _, test_loader = get_dataloaders(DATA_DIR, batch_size=64)

    print("\nRunning evaluation...")
    labels, preds = get_all_predictions(model, test_loader, device)

    overall_acc = (labels == preds).mean() * 100
    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")

    plot_confusion_matrix(labels, preds, SAVE_DIR)
    print_classification_report(labels, preds, SAVE_DIR)
    plot_per_emotion_accuracy(labels, preds, SAVE_DIR)

    print("\nEvaluation complete! Check models/ folder for all graphs.")