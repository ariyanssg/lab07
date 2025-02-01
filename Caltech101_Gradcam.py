import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# =======================
# Part 1: Data Loading and Preprocessing
# =======================

# Define data augmentation and normalization transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128 pixels
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomVerticalFlip(),  # Randomly flip images vertically
    transforms.RandomRotation(30),  # Randomly rotate images up to 30 degrees
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Adjust image color properties
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random affine transformations
    transforms.RandomGrayscale(p=0.2),  # Convert to grayscale with 20% probability
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load the Caltech-101 dataset
# Note: Update 'path_to_caltech101' to the dataset's root directory
dataset = datasets.ImageFolder(root='path_to_caltech101', transform=transform)

# Split the dataset into training (80%), validation (10%), and testing (10%)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for batching and shuffling
batch_size = 32  # Define the batch size
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# =======================
# Part 2: Model Definition
# =======================

# Load a pre-trained ResNet50 model and modify it for 101 classes
model = models.resnet50(pretrained=True)  # Load the ResNet50 model pre-trained on ImageNet
model.fc = nn.Linear(2048, 101)  # Replace the final fully connected layer for 101 classes

# =======================
# Part 3: Training Setup
# =======================

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with initial learning rate of 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
model = model.to(device)  # Move the model to the appropriate device

# =======================
# Part 4: Training the Model
# =======================

num_epochs = 10  # Number of epochs to train

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        # Move images and labels to the appropriate device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()  # Clear gradients from the previous step
        outputs = model(images)  # Get model predictions
        loss = criterion(outputs, labels)  # Compute loss

        # Backward pass
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model weights

        running_loss += loss.item() * images.size(0)  # Accumulate loss

    # Compute average training loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # =======================
    # Validate the model
    # =======================
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move images and labels to the appropriate device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)  # Accumulate validation loss
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            total += labels.size(0)  # Total number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Compute average validation loss and accuracy
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# =======================
# Part 5: Evaluating the Model on Test Data
# =======================

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            # Move images and labels to the appropriate device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class

            y_true.extend(labels.cpu().numpy())  # Move labels to CPU and collect
            y_pred.extend(predicted.cpu().numpy())  # Move predictions to CPU and collect

    # Compute and print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Print classification report
    class_names = dataset.classes  # Get class names from the dataset
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

evaluate_model(model, test_loader)

# =======================
# Part 6: Grad-CAM Visualization
# =======================

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Define the target layer for Grad-CAM (last layer of ResNet50)
target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layer=target_layer, use_cuda=torch.cuda.is_available())

# Visualize Grad-CAM for a few test images
model.eval()
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        if i == 5:  # Visualize Grad-CAM for 5 images
            break

        images = images.to(device)
        grayscale_cam = cam(input_tensor=images, target_category=labels[0].item())

        # Convert image to NumPy format and overlay Grad-CAM
        cam_image = show_cam_on_image(images[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam)
        plt.imshow(cam_image)
        plt.title(f"Grad-CAM for Class {labels[0].item()}")
        plt.show()
