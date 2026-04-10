import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def show_sample_images(data_dir, split='train'):
    """
    Shows one sample image per emotion from the dataset.
    """
    folder = os.path.join(data_dir, split)
    fig, axes = plt.subplots(1, 7, figsize=(14, 3))
    fig.suptitle(f'Sample images from {split} set', fontsize=14)

    for i, emotion in enumerate(EMOTIONS):
        emotion_folder = os.path.join(folder, emotion)
        images = os.listdir(emotion_folder)
        random_img = random.choice(images)
        img_path = os.path.join(emotion_folder, random_img)

        img = mpimg.imread(img_path)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(emotion, fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png')  # save to project folder
    plt.show()
    print("Saved as sample_images.png")


def show_emotion_counts(data_dir, split='train'):
    """
    Shows how many images exist per emotion (checks for imbalance).
    """
    folder = os.path.join(data_dir, split)
    counts = {}

    for emotion in EMOTIONS:
        emotion_folder = os.path.join(folder, emotion)
        counts[emotion] = len(os.listdir(emotion_folder))

    # Print counts
    print(f"\nImage counts per emotion ({split} set):")
    print("-" * 30)
    for emotion, count in counts.items():
        bar = "█" * (count // 200)
        print(f"{emotion:<10} {count:>5}  {bar}")

    return counts


if __name__ == '__main__':
    DATA_DIR = 'data'
    show_emotion_counts(DATA_DIR, split='train')
    show_sample_images(DATA_DIR, split='train')