import kagglehub
import shutil
import os

# 1. Descargar dataset
path = kagglehub.dataset_download("tr1gg3rtrash/yoga-posture-dataset")
print("Dataset descargado en:", path)

# 2. Carpeta destino dentro del proyecto
project_data_path = os.path.join("..", "data", "yoga_posture_dataset")

# 3. Si ya existe, la borramos para poner la nueva versión
if os.path.exists(project_data_path):
    shutil.rmtree(project_data_path)

# 4. Copiar dataset descargado a la carpeta del proyecto
shutil.copytree(path, project_data_path)

print("Dataset copiado a:", project_data_path)
