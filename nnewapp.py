from fastai import *
from fastai.vision.widgets import *
from fastai.vision.all import *

import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image




x = "best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
model = model.autoshape()





class Predict:
    def __init__(self, filename):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
 
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])

        

    def display_output(self):
        result = model(uploaded_file, size=640)
        result.save() 
        result.render()

        st.image(result.render())
 

if __name__=='__main__':
    predictor = Predict(x)
