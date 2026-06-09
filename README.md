# Waste Classification System

An iOS app that classifies waste items into recycling categories using a PyTorch model served via a Flask API.

Built as an A-level Computer Science project. The system takes a photo of a waste item and returns a category with a confidence score, aiming to reduce recycling contamination caused by manual sorting errors.

## How it works

1. User takes or uploads a photo in the iOS app
2. The app sends the image to a local Flask server via HTTP POST
3. The server pre-processes the image and runs inference using a fine-tuned ResNet-18 model
4. The predicted category and confidence score are returned as JSON and displayed in the app

## Categories

Cardboard · Glass · Metal · Paper · Plastic · Trash

## Stack

- **Model**: ResNet-18 (transfer learning from ImageNet, fine-tuned on TrashNet dataset)
- **Backend**: Python, PyTorch, Flask
- **Frontend**: Swift / SwiftUI (iOS)
- **Dataset**: TrashNet (2,527 labelled images across 6 classes)

## Results

Evaluated on 1,103 images from a held-out test set (Kaggle garbage classification dataset, not used in training):

- **Overall accuracy: 72.6%**
- Response time: under 2 seconds per classification
- All function tests passed (image upload, camera capture, error handling, offline behaviour)

The 85% accuracy target was not met. Main limiting factors were the small, controlled training dataset and limited hardware for training. Retraining on a more diverse dataset with varied lighting and backgrounds is the primary improvement path.

## Architecture

```
iOS App (SwiftUI)
    └── HTTP POST (multipart/form-data)
            └── Flask Server (server.py)
                    └── ResNet-18 model (model.py)
```

The model is loaded into memory once at server startup. Each request pre-processes the image (resize to 224×224, normalise using ImageNet statistics) before inference.

## Project structure

```
├── model.py          # Model architecture, data transforms, DataLoader setup
├── train.py          # Training loop with early stopping and best-model saving
├── server.py         # Flask API endpoint (/classify)
├── testmodel.py      # Batch accuracy evaluation script
└── wasteClassificationUI/   # Swift iOS app
    ├── ContentView.swift
    ├── ImagePicker.swift
    └── ClassificationResponse.swift
```

## Running the server

```bash
pip install torch torchvision flask pillow
python server.py
```

Server runs on `http://0.0.0.0:8000`. The iOS app connects to the local network IP.

## Training

```bash
python train.py
```

Trains ResNet-18 on TrashNet with early stopping (patience=3), saves best model to `model_baseline_resnet18.pth`.

## Design decisions

- **Transfer learning**: ResNet-18 pre-trained on ImageNet used as base — practical given limited hardware and dataset size
- **Client-server architecture**: keeps the model off the device, simplifying iOS app complexity
- **Early stopping**: prevents overfitting on the small dataset
- **Confidence score**: shown to user so they know when to trust or override the prediction

## Known limitations

- Accuracy drops on real-world photos with cluttered backgrounds or unusual lighting
- App requires server to be running on the same network (no offline mode)
- Single item classification only — no batch or multi-object support
