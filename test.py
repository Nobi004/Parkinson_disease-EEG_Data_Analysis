import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")