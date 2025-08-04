import torch
import numpy as np
import requests
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import cv2

# # Load an image
# img = Image.open(
#     requests.get(
#         "/d/hanr/dataset/her2st-master/data/ST-imgs/F/F1/FHE_BT23810_D2.jpg",
#         stream=True
#     ).raw
# )
# imgs = Image.open("/d/hanr/dataset/DLPFC/151508/spatial/full_image.tif")
full_image = cv2.imread("/d/hanr/dataset/DLPFC/151508/spatial/full_image.tif")
full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
patches = []
for x, y in adata.obsm['spatial']:
    patches.append(full_image[y-img_size:y+img_size, x-img_size:x+img_size])
patches = np.array(patches)
# imgs = np.load("/d/hanr/dataset/her2st-master/data/ST-imgs/F/F1/FHE_BT23810_D2.jpg")
# for i, img in enumerate(imgs):
#     print('extracting img ', i)
#     if img.shape[0] != 224:
#         img = cv2.resize(img, (224, 224))
#     img = Image.fromarray(img.astype(np.uint8))
#     img.convert('RGB')
# Load phikon-v2
processor = AutoImageProcessor.from_pretrained("/d/hanr/phikon-v2")
model = AutoModel.from_pretrained("/d/hanr/phikon-v2")
model.eval()

# Process the image
inputs = processor(imgs, return_tensors="pt")
inputs_1 = {"pixel_values":torch.from_numpy(imgs_1).squeeze(dim=0)}
print(inputs["pixel_values"].shape,inputs_1["pixel_values"].shape)

# Get the features
with torch.inference_mode():
    outputs = model(**inputs)
    print(outputs.last_hidden_state,outputs.last_hidden_state.shape)
    features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape
    print(features,features.shape)
    np.save(f"/d/hanr/EMSC/results/img_feat_phi/"+"151508", features.numpy())
# assert features.shape == (1, 1024)