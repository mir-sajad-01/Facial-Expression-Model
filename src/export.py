import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, get_device

MODEL_PATH = '../models/best_model.pth'
EXPORT_DIR = '../models'


def load_trained_model(model_path, device):
    model = build_model(pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded — epoch {checkpoint['epoch']}, "
          f"val_acc {checkpoint['val_acc']:.2f}%")
    return model


def export_onnx(model, export_dir, device):
    path        = os.path.join(export_dir, 'model.onnx')
    dummy_input = torch.randn(1, 1, 48, 48).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=11,
        input_names=['face_image'],
        output_names=['emotion_scores'],
        dynamic_axes={
            'face_image':     {0: 'batch_size'},
            'emotion_scores': {0: 'batch_size'}
        }
    )
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"ONNX model saved — {size_mb:.1f} MB")


def export_torchscript(model, export_dir, device):
    path        = os.path.join(export_dir, 'model_scripted.pt')
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    scripted    = torch.jit.trace(model, dummy_input)
    scripted.save(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"TorchScript model saved — {size_mb:.1f} MB")


if __name__ == '__main__':
    device = get_device()

    print("Loading model...")
    model = load_trained_model(MODEL_PATH, device)

    os.makedirs(EXPORT_DIR, exist_ok=True)

    print("\nExporting to ONNX...")
    export_onnx(model, EXPORT_DIR, device)

    print("\nExporting to TorchScript...")
    export_torchscript(model, EXPORT_DIR, device)

    print("\nAll exports complete!")
    for f in ['model.onnx', 'model_scripted.pt']:
        path = os.path.join(EXPORT_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024*1024)
            print(f"  {f} — {size:.1f} MB")