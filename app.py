import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image




x = "best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
model = model.autoshape()
uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])




class Predict:
    def __init__(self, x):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
        self.display_output()
     
            
            
  

    def display_output(self):
        # Inference
        
        result = model(uploaded_file, size=640)
        results.render()  # updates results.imgs with boxes and labels
        # includes NMS
        result.print()  
        result.save() 
        st.image("results/uploaded_file")
 
predictor = Predict(x)




