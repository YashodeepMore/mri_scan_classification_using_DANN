import base64
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# ---------- Lazy Load Model ----------
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model('brain_tumor_model.keras')
    return model

IMG_SIZE = (224, 224)
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

LAST_CONV_LAYER = "conv5_block3_out"

# ---------- Request Schema ----------
class ImageRequest(BaseModel):
    image_base64: str

# ---------- Preprocess ----------
def preprocess(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ---------- Grad-CAM ----------
def get_gradcam_heatmap(model, img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # avoid division by zero crash
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros((224, 224))

    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()

# ---------- Convert image to base64 ----------
def image_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# ---------- Overlay ----------
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
        # Decode base64
        image_bytes = base64.b64decode(request.image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        img_array = preprocess(img)

        model = get_model()

        preds = model.predict(img_array)
        pred_class = class_labels[np.argmax(preds)]
        confidence = float(np.max(preds))

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