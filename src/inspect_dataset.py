import os

# Ruta al dataset dentro del proyecto
dataset_path = "../data/yoga_posture_dataset"

# Listo todas las carpetas del dataset (cada carpeta es una clase)
classes = os.listdir(dataset_path)
print("Clases encontradas:", classes)

# Recorro cada clase para contar cuántas imágenes tiene
for c in classes:
    class_path = os.path.join(dataset_path, c)

    # Verifico que sea una carpeta y no un archivo
    if os.path.isdir(class_path):

        # Cuento solo los archivos de imagen válidos
        count = len([
            f for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        # Imprimo el nombre de la clase y cuántas imágenes tiene
        print(f"Clase: {c} - Imágenes: {count}")
