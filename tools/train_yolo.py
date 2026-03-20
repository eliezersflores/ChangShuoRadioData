# =========================================================
# YOLO TRAINING SCRIPT
# =========================================================
# This script trains a YOLO model using the CSRD dataset after
# conversion to YOLO-compatible image and label formats.
#
# It defines the main training configuration, removes cached
# label files from previous runs, loads the selected YOLO model,
# and starts the training process using the dataset configuration
# stored in the corresponding YAML file.
#
# This script was developed by Prof. Eliezer Soares Flores
# (Google Scholar Profile: https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao).
#
# If you have any questions, please feel free to contact me at
# eliezersflores@gmail.com
# =========================================================

import os
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
window_name = 'hamming'
granularity = 'family9'      # 'binary', 'family9', or 'original100'
resize_mode = 'offline'      # 'offline' or 'yolo'
target_size = 640            # used when resize_mode = 'offline'

model_name = 'yolo26m.pt'    # 'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', or 'yolo26x.pt'
epochs = 100
imgsz = 640
batch = 16

# =========================================================
# RESIZE TAG
# =========================================================
if resize_mode == 'offline':
    resize_tag = f'{resize_mode}{target_size}'
elif resize_mode == 'yolo':
    resize_tag = resize_mode
else:
    raise ValueError(f'Invalid resize_mode: {resize_mode}')

# =========================================================
# PATHS
# =========================================================
base_dir = os.path.join(
    '..', 'data', 'CSRD2025',
    f'yolo_{window_name}_{granularity}_{resize_tag}'
)

labels_dir = os.path.join(base_dir, 'labels')

yaml_name = f'config_yolo_{window_name}_{granularity}_{resize_tag}.yaml'
config = os.path.join(os.getcwd(), yaml_name)

# =========================================================
# REMOVE .cache FILES
# =========================================================
for root, _dirs, files in os.walk(labels_dir):
    for file in files:
        if file.endswith('.cache'):
            cache_path = os.path.join(root, file)
            print(f'Removing cache file: {cache_path}')
            os.remove(cache_path)

# =========================================================
# TRAINING
# =========================================================
model = YOLO(model_name)

model.train(
    data=config,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    optimizer='AdamW',
    lr0=0.001,
    amp=False,
    mosaic=1.0,
    mixup=0.1
)

print('Training completed ✅')
