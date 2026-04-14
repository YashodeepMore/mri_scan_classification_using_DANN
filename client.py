import requests
import base64

API_URL = "http://127.0.0.1:8000/predict"  # change if deployed

# Image path
image_path = "Tr-pi_1396.jpg"

# Convert image to base64
with open(image_path, "rb") as img_file:
    encoded_string = base64.b64encode(img_file.read()).decode("utf-8")

# Request payload
payload = {
    "image_base64": encoded_string
}

# Send request
response = requests.post(API_URL, json=payload)

# Parse response
data = response.json()

print("Prediction:", data["prediction"])
print("Confidence:", data["confidence"])

# Save returned images
with open("gradcam_overlay.png", "wb") as f:
    f.write(base64.b64decode(data["gradcam_overlay"]))

with open("heatmap.png", "wb") as f:
    f.write(base64.b64decode(data["heatmap"]))

print("Saved Grad-CAM overlay and heatmap images.")