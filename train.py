#train.py

import torch
from torch import nn, optim

from model import (
    projectRoot,
    dataDir,
    buildModel,
    getDataloaders,
)

# Training config
modelPath = projectRoot / "model_baseline_resnet18.pth"
numEpochs = 15          # maximum number of passes through the training data
learningRate = 0.001
patience = 3            # early stopping patience (epochs with no improvement)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainOneEpoch(model, trainLoader, lossFunction, optimizer):
    """
    Train the model for one full pass through the training data.
    Returns the average training loss and training accuracy.
    """
    model.train()

    runningLoss = 0.0
    correctPredictions = 0
    totalImages = 0

    for images, labels in trainLoader:
        images = images.to(device)
        labels = labels.to(device)

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
    """
    Evaluate the model on the validation data.
    Returns the average validation loss and validation accuracy.
    """
    model.eval()

    runningLoss = 0.0
    correctPredictions = 0
    totalImages = 0

    with torch.no_grad():
        for images, labels in valLoader:
            images = images.to(device)
            labels = labels.to(device)

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
    print(f"Project root: {projectRoot}")
    print(f"Data dir: {dataDir}")
    print(f"Model path: {modelPath}")
    print(f"Using device: {device}")

    # 1. Load data
    trainLoader, valLoader, classNames = getDataloaders()
    print("Class names:", classNames)

    # 2. Build model
    model = buildModel().to(device)
    print("Model loaded OK")
    print("Final layer:", model.fc)

    # 3. Set up loss function and optimiser
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    # 4. Training loop with early stopping and saving best model
    bestValAccuracy = 0.0
    epochsWithoutImprovement = 0

    for epoch in range(1, numEpochs + 1):
        print(f"\nEpoch {epoch}/{numEpochs}")

        trainLoss, trainAcc = trainOneEpoch(
            model, trainLoader, lossFunction, optimizer
        )
        valLoss, valAcc = evaluateModel(
            model, valLoader, lossFunction
        )

        print(f"  Training   - loss: {trainLoss:.4f}, accuracy: {trainAcc:.3f}")
        print(f"  Validation - loss: {valLoss:.4f}, accuracy: {valAcc:.3f}")

        # Check if this is the best validation accuracy so far
        if valAcc > bestValAccuracy:
            bestValAccuracy = valAcc
            epochsWithoutImprovement = 0

            # save the current best model
            torch.save(model.state_dict(), modelPath)
            print(f"  New best model saved with validation accuracy {bestValAccuracy:.3f}")
        else:
            epochsWithoutImprovement += 1
            print(f"  No improvement on validation for {epochsWithoutImprovement} epoch(s).")

        # Early stopping check
        if epochsWithoutImprovement >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining finished. Best validation accuracy: {bestValAccuracy:.3f}")
    print(f"Best model saved to: {modelPath}")


if __name__ == "__main__":
    main()
