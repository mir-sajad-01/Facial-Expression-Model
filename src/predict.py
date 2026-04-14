import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, get_device

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOTION_EMOJIS = {
    'Angry':    '😠',
    'Disgust':  '🤢',
    'Fear':     '😨',
    'Happy':    '😊',
    'Neutral':  '😐',
    'Sad':      '😢',
    'Surprise': '😲'
}


def load_model(model_path, device):
    model = build_model(pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded — val_acc {checkpoint['val_acc']:.2f}%")
    return model


def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert('RGB')

    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    img_np  = np.array(image)
    gray_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces   = face_cascade.detectMultiScale(
        gray_np, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad      = int(0.1 * w)
        x1, y1   = max(0, x-pad), max(0, y-pad)
        x2, y2   = min(img_np.shape[1], x+w+pad), min(img_np.shape[0], y+h+pad)
        face_img = image.crop((x1, y1, x2, y2))
        print(f"Face detected at ({x}, {y}, {w}, {h})")
    else:
        face_img = image
        print("No face detected — using full image")

    tensor = transform(face_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    return probs


def show_result(image_path, probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Show image
    img = Image.open(image_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=13)

    # Show prediction bars
    colors = ['#ff6b6b' if p == max(probs) else '#74b9ff' for p in probs]
    bars   = ax2.barh(EMOTIONS, probs * 100, color=colors, edgecolor='black')
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title(f'Prediction: {EMOTIONS[probs.argmax()]} '
                  f'{EMOTION_EMOJIS[EMOTIONS[probs.argmax()]]}', fontsize=13)
    ax2.set_xlim(0, 100)

    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150)
    plt.show()
    print(f"Result saved as prediction_result.png")


if __name__ == '__main__':
    MODEL_PATH = '../models/best_model.pth'
    device     = get_device()

    print("Loading model...")
    model = load_model(MODEL_PATH, device)

    # ── Change this to your image path ───────────────────────────────────
    IMAGE_PATH = input("Enter image path: ").strip()

    if not os.path.exists(IMAGE_PATH):
        print(f"Image not found: {IMAGE_PATH}")
        exit()

    print("Predicting...")
    probs = predict_image(IMAGE_PATH, model, device)

    print(f"\nResults:")
    print("=" * 35)
    for emotion, prob in sorted(zip(EMOTIONS, probs),
                                key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"{emotion:<10} {prob*100:5.1f}%  {bar}")

    print(f"\nPrediction: {EMOTIONS[probs.argmax()]} "
          f"{EMOTION_EMOJIS[EMOTIONS[probs.argmax()]]}")

    show_result(IMAGE_PATH, probs)