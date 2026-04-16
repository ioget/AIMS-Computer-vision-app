"""
Intel Image Classifier — Flask Application
Supports PyTorch (.pth) and Keras/TensorFlow (.keras) models
"""

import os
import io
import math
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB

# Model paths — models/ folder inside flask_app/
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'rosly_mamekem_model.pth')
KERAS_MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'rosly_mamekem_model.keras')

CLASSES  = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE = (64, 64)   # model was trained on 64×64 images

# ── Model architecture (mirrors the saved state dict) ────────────────────────
#
#  features:
#    0  Conv2d(3→32, 3×3, pad=1)   |  4  Conv2d(32→64, 3×3, pad=1)
#    1  BatchNorm2d(32)             |  5  BatchNorm2d(64)
#    2  ReLU                        |  6  ReLU
#    3  MaxPool2d(2)                |  7  MaxPool2d(2)
#    8  Conv2d(64→128, 3×3, pad=1) | 12  Conv2d(128→256, 3×3, pad=1)
#    9  BatchNorm2d(128)            | 13  BatchNorm2d(256)
#   10  ReLU                        | 14  ReLU
#   11  MaxPool2d(2)                | 15  MaxPool2d(2)
#
#  With 64×64 input → 4 MaxPool2d(2) → 4×4 feature map
#  → flatten → 256*4*4 = 4096
#
#  classifier:
#    0  Flatten
#    1  Linear(4096 → 512)
#    2  ReLU
#    3  Dropout(0.5)
#    4  Linear(512 → 6)

def build_model():
    import torch.nn as nn
    return nn.Sequential(
        # features block (wrapped to match state-dict keys)
        *[]  # defined inline via a module below
    )

class IntelCNN(object):
    """Lazy wrapper — imported only when torch is available."""

    @staticmethod
    def create():
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),     # 0
                    nn.BatchNorm2d(32),                  # 1
                    nn.ReLU(inplace=True),               # 2
                    nn.MaxPool2d(2, 2),                  # 3
                    nn.Conv2d(32, 64, 3, padding=1),     # 4
                    nn.BatchNorm2d(64),                  # 5
                    nn.ReLU(inplace=True),               # 6
                    nn.MaxPool2d(2, 2),                  # 7
                    nn.Conv2d(64, 128, 3, padding=1),    # 8
                    nn.BatchNorm2d(128),                 # 9
                    nn.ReLU(inplace=True),               # 10
                    nn.MaxPool2d(2, 2),                  # 11
                    nn.Conv2d(128, 256, 3, padding=1),   # 12
                    nn.BatchNorm2d(256),                 # 13
                    nn.ReLU(inplace=True),               # 14
                    nn.MaxPool2d(2, 2),                  # 15
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),                        # 0
                    nn.Linear(4096, 512),               # 1
                    nn.ReLU(inplace=True),               # 2
                    nn.Dropout(0.5),                    # 3
                    nn.Linear(512, 6),                  # 4
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        return _Net()

# ── Load models at startup ────────────────────────────────────────────────────

pytorch_model = None
keras_model   = None


def load_pytorch():
    global pytorch_model
    try:
        import torch
        state_dict = torch.load(PYTORCH_MODEL_PATH, map_location='cpu', weights_only=False)
        model = IntelCNN.create()
        model.load_state_dict(state_dict)
        model.eval()
        pytorch_model = model
        print("[OK] PyTorch model loaded.")
    except Exception as exc:
        print(f"[WARN] PyTorch model failed to load: {exc}")


def load_keras():
    global keras_model
    try:
        import tensorflow as tf
        keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        print("[OK] Keras model loaded.")
    except ImportError:
        print("[WARN] TensorFlow not installed — Keras model unavailable.")
    except Exception as exc:
        print(f"[WARN] Keras model failed to load: {exc}")


load_pytorch()
load_keras()

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_pytorch(image_bytes: bytes):
    from torchvision import transforms
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def preprocess_keras(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ────────────────────────────────────────────────────────────────────

RESULT_DIR = os.path.join(BASE_DIR, 'result')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result/<path:filename>')
def result_file(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image received.'}), 400

    file        = request.files['image']
    model_type  = request.form.get('model', 'pytorch')
    image_bytes = file.read()

    try:
        if model_type == 'pytorch':
            if pytorch_model is None:
                return jsonify({'error': 'PyTorch model not available.'}), 500

            import torch
            import torch.nn.functional as F

            tensor = preprocess_pytorch(image_bytes)
            with torch.no_grad():
                outputs = pytorch_model(tensor)
                probs   = F.softmax(outputs, dim=1)
                conf_t, pred_t = torch.max(probs, dim=1)
                confidence = float(conf_t.item())
                class_idx  = int(pred_t.item())
            model_accuracy = 0.9214

        else:  # tensorflow / keras
            if keras_model is None:
                return jsonify({'error': 'Keras model not available (TensorFlow not installed).'}), 500

            arr        = preprocess_keras(image_bytes)
            preds      = keras_model.predict(arr, verbose=0)
            class_idx  = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))
            model_accuracy = 0.9047

        loss = -math.log(max(confidence, 1e-8))

        return jsonify({
            'prediction': CLASSES[class_idx],
            'confidence': round(confidence, 4),
            'metrics': {
                'accuracy': model_accuracy,
                'loss':     round(loss, 4),
            },
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
