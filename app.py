import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image




x = "best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)




class Predict:
    def __init__(self, x):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
     
            
            
  

    def display_output(self):
        # Inference
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        result = model(uploaded_file, size=640)  # includes NMS
        result.print()  
        result.save() 
        st.image("results/uploaded_file")
 
predictor = Predict(x)




