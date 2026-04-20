import requests
import base64

# API_URL = "https://mri-scan-classification-using-dann-2.onrender.com/predict"
API_URL = "http://127.0.0.1:8000/predict"

# image_path = "images\Tr-pi_1396.jpg"
image_path = "images/gg (707).jpg"
# image_path = "images/a1977.jpg"
# image_path= "images/image(13).jpg"
# image_path="images/image(83).jpg"
# image_path="images/Te-gl_389.jpg"

# Encode image
with open(image_path, "rb") as img_file:
    encoded_string = base64.b64encode(img_file.read()).decode("utf-8")

payload = {
    "image_base64": encoded_string
}

response = requests.post(API_URL, json=payload)

print("STATUS:", response.status_code)

# 🔴 Handle non-JSON safely
if response.status_code != 200:
    print("ERROR RESPONSE:")
    print(response.text)
    exit()

try:
    data = response.json()
except Exception:
    print("NOT JSON RESPONSE:")
    print(response.text)
    exit()

print("FULL RESPONSE:", data)

# 🔴 Safe access
if "prediction" not in data:
    print("API ERROR:", data)
    exit()

print("Prediction:", data["prediction"])
print("Confidence:", data["confidence"])

# Save images
with open("gradcam_overlay.png", "wb") as f:
    f.write(base64.b64decode(data["gradcam_overlay"]))

with open("heatmap.png", "wb") as f:
    f.write(base64.b64decode(data["heatmap"]))

print("Saved Grad-CAM overlay and heatmap images.")