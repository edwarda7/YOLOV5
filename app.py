

import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='/best.pt')

from PIL import Image

# Images
img1 = Image.open('/content/train/images/BloodImage_00002_jpg.rf.8abfeee935647b859f251b4ea8fb05b6.jpg')

results = model(img1, size=320)

# Inference
results = model(img1, size=640)  # includes NMS
results.print()  
results.save()  # or .show()

# Data
