import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# The 7 emotions — order must match folder names
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Transforms ──────────────────────────────────────────────────────────────
# Training: augment images so model sees variety
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure grayscale
    transforms.Resize((48, 48)),                  # FER2013 is 48x48
    transforms.RandomHorizontalFlip(p=0.5),       # flip face left/right
    transforms.RandomRotation(degrees=10),         # slight rotation
    transforms.ColorJitter(brightness=0.2,         # vary brightness
                           contrast=0.2),
    transforms.ToTensor(),                         # convert to tensor
    transforms.Normalize(mean=[0.5],               # normalize pixels
                         std=[0.5])
])

# Testing: no augmentation — just clean resize and normalize
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


# ── Dataset class ────────────────────────────────────────────────────────────
class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to train/ or test/ folder
        transform: image transformations to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (image_path, label) tuples

        # Walk through each emotion folder and collect image paths
        for label, emotion in enumerate(EMOTIONS):
            emotion_folder = os.path.join(root_dir, emotion)
            if not os.path.exists(emotion_folder):
                print(f"Warning: folder not found: {emotion_folder}")
                continue
            for img_file in os.listdir(emotion_folder):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(emotion_folder, img_file)
                    self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # open as grayscale

        if self.transform:
            image = self.transform(image)

        return image, label


# ── DataLoaders ──────────────────────────────────────────────────────────────
def get_dataloaders(data_dir, batch_size=64):
    """
    Returns train and test DataLoaders ready for training.
    data_dir: path to your data/ folder
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir  = os.path.join(data_dir, 'test')

    train_dataset = FERDataset(train_dir, transform=train_transform)
    test_dataset  = FERDataset(test_dir,  transform=test_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,       # shuffle every epoch
                              num_workers=0)      # 0 = safe for Windows

    test_loader  = DataLoader(test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)

    return train_loader, test_loader


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_DIR = 'data'
    train_loader, test_loader = get_dataloaders(DATA_DIR)

    # Peek at one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")   # should be [64, 1, 48, 48]
    print(f"Labels      : {labels[:8]}")     # first 8 emotion numbers
    print(f"Emotion names: {[EMOTIONS[l] for l in labels[:8]]}")