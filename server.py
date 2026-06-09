#server.py

from pathlib import Path
from io import BytesIO

import torch
from PIL import Image
from flask import Flask, request, jsonify

from model import projectRoot, dataDir, evalTransform, buildModel
from torchvision import datasets


app = Flask(__name__)

modelPath = projectRoot / "model_baseline_resnet18.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getClassNames():
    tempDataset = datasets.ImageFolder(root=dataDir, transform=evalTransform)
    return tempDataset.classes


def loadTrainedModel(classCount: int):
    if not modelPath.exists():
        raise FileNotFoundError(f"Model file not found at: {modelPath}")

    model = buildModel(customNumClasses=classCount).to(device)
    stateDict = torch.load(modelPath, map_location=device)
    model.load_state_dict(stateDict)
    model.eval()
    return model


classNames = getClassNames()
model = loadTrainedModel(classCount=len(classNames))

print("Server starting...")
print("Classes:", classNames)
print("Model loaded OK (trained weights applied).")
print("Using device:", device)


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field name should be 'image')"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        imageBytes = file.read()
        image = Image.open(BytesIO(imageBytes)).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    imageTensor = evalTransform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(imageTensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)

    predictedIndex = int(torch.argmax(probabilities).item())
    predictedClass = classNames[predictedIndex]
    confidence = float(probabilities[predictedIndex].item())

    return jsonify({
        "predictedClass": predictedClass,
        "confidence": round(confidence, 3)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
