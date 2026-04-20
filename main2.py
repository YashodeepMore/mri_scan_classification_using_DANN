import base64
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# ---------- Custom Layer ----------
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self, lambda_=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def call(self, x):
        @tf.custom_gradient
        def reverse(x):
            def grad(dy):
                return -self.lambda_ * dy
            return x, grad
        return reverse(x)

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_": self.lambda_})
        return config

# ---------- Load Model ----------
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(
            "brain_tumor_model.keras",
            custom_objects={"GradientReversalLayer": GradientReversalLayer}
        )
    return model

# ---------- Config ----------
IMG_SIZE = (224, 224)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ---------- Request ----------
class ImageRequest(BaseModel):
    image_base64: str

# ---------- Preprocess ----------
def preprocess(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---------- Find Last Conv Layer ----------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found")

# ---------- Fix prediction shape ----------
def fix_shape(preds):
    preds = preds[0] if isinstance(preds, list) else preds
    preds = np.array(preds)

    # 🔥 critical fix
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)

    return preds

# ---------- Grad-CAM ----------
def get_gradcam_heatmap(model, img_array):
    last_conv = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv).output,
            model.output[0]
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)

        # 🔥 fix shape
        preds = tf.convert_to_tensor(preds)
        if len(preds.shape) == 1:
            preds = tf.expand_dims(preds, axis=0)

        if preds.shape[-1] == 0:
            raise ValueError(f"Invalid prediction shape: {preds.shape}")

        pred_index = int(tf.argmax(preds[0]))
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros((224, 224))

    heatmap /= max_val
    return heatmap.numpy()

# ---------- Utils ----------
def image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def create_overlay(original_img, heatmap):
    img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMG_SIZE)

    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    return overlay, heatmap_color

# ---------- API ----------
@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_array = preprocess(img)

        model = get_model()

        preds = model.predict(img_array)

        # 🔥 FIXED handling
        label_preds = fix_shape(preds)

        if label_preds.shape[-1] == 0:
            return {"error": f"Invalid prediction shape: {label_preds.shape}"}

        pred_index = int(np.argmax(label_preds[0]))
        pred_class = class_labels[pred_index]
        confidence = float(label_preds[0][pred_index])

        # Grad-CAM
        heatmap = get_gradcam_heatmap(model, img_array)
        overlay, heatmap_img = create_overlay(img, heatmap)

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "gradcam_overlay": image_to_base64(overlay),
            "heatmap": image_to_base64(heatmap_img)
        }

    except Exception as e:
        return {"error": str(e)}