import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

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

def build_model():
    model = models.mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(
        in_channels=1, out_channels=32,
        kernel_size=3, stride=2, padding=1, bias=False
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 7)
    )
    return model

device    = torch.device('cpu')
model     = build_model()
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded — val_acc {checkpoint['val_acc']:.2f}%")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


def predict_emotion(image):
    if image is None:
        return {}

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Convert to numpy for face detection
    img_np    = np.array(image.convert('RGB'))
    gray_np   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_np, scaleFactor=1.1,
        minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        # Crop to first detected face
        x, y, w, h = faces[0]
        # Add small padding
        pad = int(0.1 * w)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(img_np.shape[1], x + w + pad)
        y2  = min(img_np.shape[0], y + h + pad)
        face_img = image.crop((x1, y1, x2, y2))
    else:
        # No face detected — use full image
        face_img = image

    # Preprocess
    face_img = face_img.convert('RGB').convert('L')
    tensor   = transform(face_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs     = model(tensor)
        probs       = torch.softmax(outputs, dim=1)[0]
        probs_numpy = probs.cpu().numpy()

    result = {}
    for emotion, prob in zip(EMOTIONS, probs_numpy):
        label = f"{EMOTION_EMOJIS[emotion]} {emotion}"
        result[label] = float(prob)

    return result


with gr.Blocks(title="Facial Expression Recognition") as demo:

    gr.Markdown("""
    # 😊 Facial Expression Recognition
    Upload a face image and the model will detect the emotion.
    
    **Model:** MobileNetV2 trained on FER2013 (35,887 images)  
    **Accuracy:** 59.03% on test set  
    **Emotions:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise  
    **Note:** Works best with clear front-facing face photos
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Face Image",
                type="pil"
            )
            submit_btn = gr.Button("Predict Emotion", variant="primary")

        with gr.Column():
            output = gr.Label(
                label="Emotion Prediction",
                num_top_classes=7
            )

    gr.Markdown("""
    ### About this project
    Built as a B.Tech final year project — trained from scratch using PyTorch and Transfer Learning.
    
    [GitHub Repository](https://github.com/mir-sajad-01/Facial-Expression-Model)
    """)

    submit_btn.click(
        fn=predict_emotion,
        inputs=image_input,
        outputs=output
    )

demo.launch(server_name="0.0.0.0", server_port=7860)