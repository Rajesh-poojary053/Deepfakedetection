from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
MODEL_PATH = "model/efficientnet_deepfake_classifier.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B4 model
model = EfficientNet.from_pretrained("efficientnet-b4")
num_features = model._fc.in_features
model._fc = nn.Linear(num_features, 2)  # Binary classification
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels
class_labels = ["Fake", "Real"]

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
   
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        print("Model Output:", outputs)  # Debugging
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, 0)
        print(f"Predicted: {predicted_class.item()}, Confidence: {confidence.item()}")  # Debugging

    return class_labels[predicted_class.item()], confidence.item()

# Web UI Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            label, confidence = predict_image(file_path)
            return render_template("index.html", filename=file.filename, label=label, confidence=confidence)
    return render_template("index.html")

# API Endpoint for Image Upload
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    label, confidence = predict_image(file_path)
    print(f"API Response: {label}, Confidence: {confidence}")  # Debugging
    
    return jsonify({"filename": file.filename, "label": label, "confidence": confidence})

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

