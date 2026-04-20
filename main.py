import base64
import gc
import io
import os

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 1. Force TensorFlow to use CPU and minimize memory bloat
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Global model containers
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once on startup
    main_model = tf.keras.models.load_model('brain_tumor_model.keras')
    
    # Pre-build the Grad-CAM model to avoid creating new objects per request
    grad_model = tf.keras.models.Model(
        [main_model.inputs],
        [main_model.get_layer("conv5_block3_out").output, main_model.output]
    )
    
    models["main"] = main_model
    models["grad"] = grad_model
    yield
    # Cleanup on shutdown
    models.clear()
    tf.keras.backend.clear_session()

app = FastAPI(lifespan=lifespan)

IMG_SIZE = (224, 224)
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

class ImageRequest(BaseModel):
    image_base64: str

def preprocess(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        # Decode image
        image_bytes = base64.b64decode(request.image_base64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = preprocess(img)

        # 2. Grad-CAM Calculation (Using the pre-built grad_model)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = models["grad"](img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Extract data for prediction return
        pred_class = class_labels[pred_index]
        confidence = float(predictions[0][pred_index])

        # Heatmap math
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Normalize heatmap
        max_val = tf.reduce_max(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (max_val if max_val > 0 else 1.0)
        heatmap_np = heatmap.numpy()

        # 3. Process Visuals (Overlay)
        img_cv = cv2.cvtColor(np.array(img.resize(IMG_SIZE)), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap_np, IMG_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

        # Encode results
        _, buf_ov = cv2.imencode('.png', overlay)
        _, buf_hm = cv2.imencode('.png', heatmap_color)

        # 4. Critical Cleanup
        del img_array, conv_outputs, predictions, heatmap, grads, img_cv
        gc.collect()

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "gradcam_overlay": base64.b64encode(buf_ov).decode('utf-8'),
            "heatmap": base64.b64encode(buf_hm).decode('utf-8')
        }

    except Exception as e:
        return {"error": str(e)}
