import torch
import streamlit as st

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='best.pt')

from PIL import Image

# Images
img1 = Image.open('BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg')


results = model(img1, size=320)

# Inference
results = model(img1, size=640)  # includes NMS
results.print()  
results.save()  # or .show()
st.image("results/BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg")
# Data
