import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import os

# ================================
# TRANSFORMACIONES PARA EL MODELO
# ================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # data augmentation básico
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================================
# CARGA DE DATASETS
# ================================
train_dataset = datasets.ImageFolder("../data/train", transform=train_transform)
val_dataset   = datasets.ImageFolder("../data/val", transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)

num_classes = len(train_dataset.classes)
print(f"Número de clases: {num_classes}")

# ================================
# MODELO PREENTRENADO
# ================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Reemplazamos la capa final para adaptarla a nuestro número de poses
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ================================
# FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ================================
# FUNCIÓN DE ENTRENAMIENTO
# ================================
def train_epoch():
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# ================================
# FUNCIÓN DE VALIDACIÓN
# ================================
def validate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# ================================
# ENTRENAMIENTO FINAL
# ================================
EPOCHS = 5
for epoch in range(EPOCHS):
    loss = train_epoch()
    acc = validate()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f} - Val Accuracy: {acc:.4f}")

# ================================
# GUARDAR MODELO
# ================================
torch.save(model.state_dict(), "../model_yoga.pth")
print("Modelo guardado como model_yoga.pth")
