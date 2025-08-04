from PIL import Image
import torch
import requests
from transformers import AutoImageProcessor, AutoModel
import numpy as np

# Load phikon-v2
processor = AutoImageProcessor.from_pretrained("/home/h3c/data/HWQ/-MAFSC/phikon-v2")
model = AutoModel.from_pretrained("/home/h3c/data/HWQ/-MAFSC/phikon-v2")
model.eval()

# Process the image
inputs = processor(image, return_tensors="pt")
# for i in range(12):
#     t2 = np.load(f"/d/wangx/patches/skin/{i}.npy")

#     inputs = {"pixel_values":torch.from_numpy(t2).squeeze(dim=0)}

# Get the features
with torch.inference_mode():
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
    np.save(f"/home/h3c/data/HWQ/-MAFSC/results/img_feat_pil", features.numpy())