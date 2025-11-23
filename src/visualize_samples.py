import matplotlib
matplotlib.use("TkAgg")

import random
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# transform simple para cargar imágenes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ruta al dataset
dataset = datasets.ImageFolder("../data/yoga_posture_dataset", transform=transform)

# mostrar 3 imágenes aleatorias
for i in range(3):
    img, label = random.choice(dataset)
    plt.imshow(img.permute(1, 2, 0))
    plt.title(dataset.classes[label])
    plt.axis("off")
    plt.show()
