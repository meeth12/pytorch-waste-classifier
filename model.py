#model.py

from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# Base paths and core constants
projectRoot = Path(__file__).resolve().parent
dataDir = projectRoot / "data" / "trashnet"

numClasses = 6
batchSize = 32


# DATA TRANSFORMS
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

# Transforms for validation / testing: no randomness, just resize + centre crop
evalTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def getDataloaders(trainRatio: float = 0.8):
    #Load the TrashNet dataset and split it into training and validation sets.
    #Returns (trainLoader, valLoader, classNames).
    if not dataDir.exists():
        raise FileNotFoundError(f"dataDir does not exist: {dataDir}")

    # Load all images using the training transform by default
    fullDataset = datasets.ImageFolder(root=dataDir,
                                       transform=trainTransform)

    classNames = fullDataset.classes
    print("Classes found:", classNames)
    print("Total images:", len(fullDataset))

    # Split into training and validation sets
    trainSize = int(trainRatio * len(fullDataset))
    valSize = len(fullDataset) - trainSize

    trainDataset, valDataset = random_split(fullDataset,
                                            [trainSize, valSize])

    # Use evalTransform for validation
    valDataset.dataset.transform = evalTransform

    trainLoader = DataLoader(trainDataset,
                             batch_size=batchSize,
                             shuffle=True)
    valLoader = DataLoader(valDataset,
                           batch_size=batchSize,
                           shuffle=False)

    print("Train size:", len(trainDataset))
    print("Val size:", len(valDataset))
    print("Train batches:", len(trainLoader))
    print("Val batches:", len(valLoader))

    return trainLoader, valLoader, classNames


def buildModel(customNumClasses: int | None = None):
    #Build a ResNet-18 model pre-trained on ImageNet and
    #replace the final fully connected layer so it outputs numClasses scores.
    effectiveNumClasses = customNumClasses if customNumClasses is not None else numClasses

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, effectiveNumClasses)
    return model
