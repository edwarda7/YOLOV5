import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image


class Predict:
    def __init__(self, filename):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='best.pt')
        if self.img is not None:
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


   
if __name__=='__main__':
    predictor = Predict(best.pt)
    






