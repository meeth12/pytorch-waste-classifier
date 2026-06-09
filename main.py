from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# Folder that contains main.py (your project root)
PROJECT_ROOT = Path(__file__).resolve().parent


# path to the TrashNet dataset
DATA_DIR = PROJECT_ROOT / "data" / "trashnet"


# where the model is saved
MODEL_PATH = PROJECT_ROOT / "model_baseline_resnet18.pth"


#Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_CLASSES = 6
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#DATA TRANSFORMS
# Transforms for training: resize, augment a bit, normalise
trainTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Transforms for validation: no randomness, just resize + centre crop
evalTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def getDataloaders():
    #Load the TrashNet dataset and split it into training and validation sets.
    #Returns (trainLoader, valLoader, classNames).

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    # Load all images using the training transform by default
    fullDataset = datasets.ImageFolder(root=DATA_DIR,
                                        transform=trainTransform)

    classNames = fullDataset.classes
    print("Classes found:", classNames)
    print("Total images:", len(fullDataset))

    # Split: 80% train, 20% validation for now
    trainSize = int(0.8 * len(fullDataset))
    valSize = len(fullDataset) - trainSize

    trainDataset, valDataset = random_split(fullDataset,
                                              [trainSize, valSize])

    # Use eval transform for validation
    valDataset.dataset.transform = evalTransform

    trainLoader = DataLoader(trainDataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    valLoader = DataLoader(valDataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False)

    print("Train size:", len(trainDataset))
    print("Val size:", len(valDataset))
    print("Train batches:", len(trainLoader))
    print("Val batches:", len(valLoader))

    return trainLoader, valLoader, classNames


#model building function
def buildModel(numClasses: int = 6):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, numClasses)
    return model

def trainOneEpoch(model, trainLoader, lossFunction, optimizer):
    #Train the model for one full pass through the training data.
    #Returns the average training loss and training accuracy.

    model.train()  # training mode

    runningLoss = 0.0
    correctPredictions = 0
    totalImages = 0

    for images, labels in trainLoader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # forward pass
        outputs = model(images)

        # compute loss
        loss = lossFunction(outputs, labels)

        # reset gradients, backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batchSize = images.size(0)
        runningLoss += loss.item() * batchSize

        # count correct predictions
        _, predicted = torch.max(outputs, dim=1)
        correctPredictions += (predicted == labels).sum().item()
        totalImages += batchSize

    averageLoss = runningLoss / totalImages
    accuracy = correctPredictions / totalImages

    return averageLoss, accuracy


def evaluateModel(model, valLoader, lossFunction):
    #Evaluate the model on the validation data.
    #Returns the average validation loss and validation accuracy.

    model.eval()  # evaluation mode

    runningLoss = 0.0
    correctPredictions = 0
    totalImages = 0

    with torch.no_grad():
        for images, labels in valLoader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = lossFunction(outputs, labels)

            batchSize = images.size(0)
            runningLoss += loss.item() * batchSize

            _, predicted = torch.max(outputs, dim=1)
            correctPredictions += (predicted == labels).sum().item()
            totalImages += batchSize

    averageLoss = runningLoss / totalImages
    accuracy = correctPredictions / totalImages

    return averageLoss, accuracy


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Using device: {DEVICE}")

    # 1. Load data
    trainLoader, valLoader, classNames = getDataloaders()

    # 2. Build model
    model = buildModel().to(DEVICE)
    print("Model loaded OK")
    print("Final layer:", model.fc)

    # 3. Set up loss function and optimiser
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        trainLoss, trainAcc = trainOneEpoch(
            model, trainLoader, lossFunction, optimizer
        )
        valLoss, valAcc = evaluateModel(
            model, valLoader, lossFunction
        )

        print(f"  Training   - loss: {trainLoss:.4f}, accuracy: {trainAcc:.3f}")
        print(f"  Validation - loss: {valLoss:.4f}, accuracy: {valAcc:.3f}")


if __name__ == "__main__":
    main()
