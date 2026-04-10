import torch
import torch.nn as nn
from torchvision import models

# Emotions — must match dataset.py order
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)  # 7


def build_model(pretrained=True, freeze_backbone=True):
    """
    Builds MobileNetV2 with a custom emotion classification head.

    pretrained:      load weights trained on ImageNet
    freeze_backbone: freeze MobileNetV2 layers (only train our head)
    """

    # ── Load pretrained MobileNetV2 ──────────────────────────────────────
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    )

    # ── Freeze backbone layers ───────────────────────────────────────────
    # We don't want to retrain what MobileNet already knows
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # ── Modify input layer for grayscale ─────────────────────────────────
    # MobileNetV2 expects 3-channel (RGB) images
    # Our faces are 1-channel (grayscale)
    # Fix: replace first conv layer to accept 1 channel
    model.features[0][0] = nn.Conv2d(
        in_channels=1,       # grayscale input
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )

    # ── Replace classifier head ──────────────────────────────────────────
    # Original MobileNetV2 head outputs 1000 classes (ImageNet)
    # We replace it with our 7-emotion head
    in_features = model.classifier[1].in_features  # 1280

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),              # dropout reduces overfitting
        nn.Linear(in_features, 256),    # 1280 → 256
        nn.ReLU(),                      # activation function
        nn.Dropout(p=0.2),
        nn.Linear(256, NUM_CLASSES)     # 256 → 7 emotions
    )

    return model


def count_parameters(model):
    """Shows how many parameters are trainable vs frozen."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"Total parameters  : {total:,}")
    print(f"Trainable         : {trainable:,}  ← we train these")
    print(f"Frozen            : {frozen:,}   ← pretrained, untouched")


def get_device():
    """Returns GPU if available, otherwise CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = get_device()
    model  = build_model(pretrained=True, freeze_backbone=True)
    model  = model.to(device)

    print("\nModel parameter summary:")
    print("-" * 40)
    count_parameters(model)

    # Test with a fake batch — simulates real training data
    print("\nRunning test forward pass...")
    fake_batch = torch.randn(4, 1, 48, 48).to(device)  # 4 fake face images
    output     = model(fake_batch)

    print(f"Input shape  : {fake_batch.shape}")   # [4, 1, 48, 48]
    print(f"Output shape : {output.shape}")        # [4, 7]
    print(f"Output sample: {output[0].detach()}")  # 7 raw scores
    print("\nModel is working correctly.")