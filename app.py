import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image







model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='best.pt')
 

img1 = Image.open('BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg')

# Inference
result = model(img1, size=640)  # includes NMS
result.print()  
result.save() 
  
            
    
 
    
st.image("results/BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg")










