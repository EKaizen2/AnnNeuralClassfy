# Importing Important Libraries
import os
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split

print("Libraries Imported")
# Re-loading MNIST from file
DATA_DIR = "."
download_dataset = False

m_train_mnist = datasets.MNIST(
    DATA_DIR,
    train=True,
    download=download_dataset,
    transform=transforms.Compose([transforms.ToTensor()]),
)
m_test_mnist = datasets.MNIST(
    DATA_DIR,
    train=False,
    download=download_dataset,
    transform=transforms.Compose([transforms.ToTensor()]),
)
print("Data sets loaded")
len(m_train_mnist), len(m_test_mnist)

# SPLITTING THE DATASET INTO TRAINING AND VALIDATION

train_ds, val_ds = random_split(m_train_mnist, [50000, 10000])
len(train_ds), len(val_ds)

batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size * 2)
test_loader = DataLoader(m_test_mnist, batch_size * 2)


#########################################################################################################
# BUILDING NEURAL CLASSIFICATION NETWORK
class AnnModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_size)

    def forward(self, x_batch):
        outputs = x_batch.reshape(-1, 784)
        outputs = self.linear4(
            F.relu(self.linear3(F.relu(self.linear2(F.relu(self.linear1(outputs))))))
        )
        return outputs


# DEFINING ACCURACY FUNCTION


def accuracy_score(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    correct_preds = torch.sum(preds == labels).item()
    total_preds = len(preds)
    accuracy = torch.tensor(correct_preds / total_preds)
    return accuracy


# DEFINING VALIDATION FUNCTIONS


def validation_step(model, batch):
    images, labels = batch
    loss = F.cross_entropy(model.forward(images), labels)
    acc = accuracy_score(model.forward(images), labels)
    return {"val_loss": loss, "val_acc": acc}


def validation_epoch_end(outputs):
    batch_losses = [x["val_loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x["val_acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}


def evaluate(model, val_loader):
    metrics = [validation_step(model, batch) for batch in val_loader]
    metrics = validation_epoch_end(metrics)
    return metrics


# DEFINING THE FIT FUNCTION


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        for batch in train_loader:
            images, labels = batch
            loss = F.cross_entropy(model.forward(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        metrics = evaluate(model, val_loader)
        print(
            "Epoch #",
            epoch + 1,
            "|",
            "Validation loss:",
            round(metrics["val_loss"], 3),
            "|",
            "Validation accuracy:",
            round(metrics["val_acc"] * 100, 3),
            "%",
        )
        with open("logs.txt", "a") as f:
            f.write(
                f'\nEpoch # {epoch + 1} | Validation loss: {round(metrics["val_loss"], 3)} | Validation accuracy: {round(metrics["val_acc"] * 100, 3)} %\n'
            )

        history.append(metrics)
    with open("logs.txt", "a") as f:
        f.write(
            f"\n -----Model Training Complete | Model Is Ready For Classification-----\n"
        )
    print(f"\n -----Model Training Complete | Model Is Ready For Classification-----\n")
    return history


# CALLING DEFINED MODEL CLASS AND FUNCTIONS TO TRAIN THE MODEL

model = AnnModel(784, 10)

history = [evaluate(model, val_loader)]
print("Metrics before training:")
with open("logs.txt", "a") as f:
    f.write(f"\n -------- MODEL TRAINING INITIATED ---------\n")
    f.write(f"\n Metrics before training:")
print(history)
with open("logs.txt", "a") as f:
    f.write(f"\n {history}")
print()
print("Training Metrics Per epochs:")
with open("logs.txt", "a") as f:
    f.write(f"\n Training Metrics Per epochs:")
history += fit(10, 0.4, model, train_loader, val_loader)


# RUNING MODEL TO TAKE IN PATH FOR IMAGE CLASSIFICATION
class Classify_image:
    # Preprocess the image
    def preprocess_image(image_path):
        # Converting input image into a grayscale
        image = Image.open(image_path).convert("L")
        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        image = transform(image)
        image = image.unsqueeze(0)
        with open("logs.txt", "a") as f:
            f.write(f"\n Image Processing Complete")
        return image

    # Path to the new image
    print(
        "\nMake sure there are no qoutation marks on the path, and the path should start from MNIST folder."
    )
    image_path = input("\nPaste Image path:")
    with open("logs.txt", "a") as f:
        f.write(f"\n Image Path Provided: {image_path}")
    if os.path.exists(image_path):
        print("\nPath verified")
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path)
        # Make predictions
        with torch.no_grad():
            output = model(preprocessed_img)
        predicted_class = torch.argmax(output).item()
        with open("logs.txt", "a") as f:
            f.write(f"\nModel Prediction is: {predicted_class:=> 30}")
        print(f"\nModel Prediction is: {predicted_class:=> 30}")
    else:
        print("\nImage does not exist or image path is wrong. Please cross check.")
        with open("logs.txt", "a") as f:
            f.write(
                f"\nImage does not exist or image path is wrong. Please cross check."
            )


Classify_image()
