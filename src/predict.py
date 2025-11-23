import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import random

# ----------------------------
# Cargar clases desde carpeta de entrenamiento
# ----------------------------
train_dir = "../data/train"
classes = sorted([
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
])

# ----------------------------
# Transformaciones
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ----------------------------
# Cargar modelo guardado
# ----------------------------
def load_model(model_path="../model/model_yoga.pth"):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ----------------------------
# Predecir una imagen
# ----------------------------
def predict(image_path):
    if not os.path.exists(image_path):
        print("La imagen no existe:", image_path)
        return

    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)

    clase_predicha = classes[predicted.item()]
    print(f"Imagen: {os.path.basename(image_path)}")
    print(f"Predicción: {clase_predicha}")

    # ----------------------------
    # Mostrar imagen con predicción
    # ----------------------------
    plt.imshow(img)
    plt.title(f"Predicción: {clase_predicha}")
    plt.axis("off")
    plt.show()

# ----------------------------
# Seleccionar imagen aleatoria de una clase
# ----------------------------
def predict_random_image_from_class(class_name):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.exists(class_path):
        print("Clase no encontrada:", class_name)
        return

    images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png"))]
    if not images:
        print("No hay imágenes en la clase:", class_name)
        return

    image_path = os.path.join(class_path, random.choice(images))
    predict(image_path)

# ----------------------------
# Ejecución desde consola
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python predict.py NombreDeClase")
        print("Ejemplo: python predict.py Bakasana")
    else:
        predict_random_image_from_class(sys.argv[1])
