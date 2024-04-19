# Importing relevant Libraries
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# ANN NEURAL IMAGE CLASSIFICATION MODEL FUNCTION
def Classify_image():
    # Set the model to evaluation mode
    class AnnModel(nn.Module):

        def __init__(self):
            super(AnnModel, self).__init__()
            self.linear1 = nn.Linear(784, 64)
            self.linear2 = nn.Linear(64, 32)
            self.linear3 = nn.Linear(32, 16)
            self.linear4 = nn.Linear(16, 10)

        def forward(self, x_batch):
            outputs = x_batch.reshape(-1, 784)
            outputs = self.linear4(
                F.relu(
                    self.linear3(F.relu(self.linear2(F.relu(self.linear1(outputs)))))
                )
            )
            return outputs

    # Loading Ann Forward feeding Neural Model
    loaded_model = AnnModel()
    loaded_model.load_state_dict(torch.load("neural_classify_model.pth"))
    loaded_model.eval()
    print("Model is Ready")

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
        return image

    # Path to the new image
    print(
        "Make sure there are no qoutation marks on the path, and the path should start from MNIST folder."
    )
    image_path = input("Paste Image path:")
    if os.path.exists(image_path):
        print("Path verified")
        # Preprocess the image
        preprocessed_img = preprocess_image(image_path)
        # Make predictions
        with torch.no_grad():
            output = loaded_model(preprocessed_img)
        predicted_class = torch.argmax(output).item()
        print(f"\nModel Prediction is: {predicted_class:=> 30}")
    else:
        print("Image does not exist or image path is wrong. Please cross check.")


Classify_image()
