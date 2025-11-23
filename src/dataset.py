# src/dataset/dataset.py

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def create_dataloaders(data_dir, batch_size=32, img_size=224):

    # Transformación principal (redimensionar + normalizar)
    base_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Aumentación SOLO en entrenamiento
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Cargar datasets desde carpetas
    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
    val_dataset   = datasets.ImageFolder(f"{data_dir}/val",   transform=base_transform)
    test_dataset  = datasets.ImageFolder(f"{data_dir}/test",  transform=base_transform)

    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

