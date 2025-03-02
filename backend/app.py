import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal, Pneumonia
model.load_state_dict(torch.load("pneumonia_classifier_final.pth", map_location=device))
model.to(device)
model.eval()

# Define Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction Function
def predict_image(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return "Pneumonia" if pred.item() == 1 else "Normal"

# API Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    prediction = predict_image(image)

    return jsonify({'prediction': prediction})

PORT = int(os.environ.get("PORT", 10000))  

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT, debug=False)