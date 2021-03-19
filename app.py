import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image
x = "best.pt"


class Predict:
    def __init__(self, x):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
     
        self.img = self.get_image_from_upload()
        self.display_output()
            
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None

    def display_output(self):
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')
        # Inference
        result = model(img1, size=640)  # includes NMS
        result.print()  
        result.save() 
        st.image("results/uploaded_file")
 
predictor = Predict(x)




