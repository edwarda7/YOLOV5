import glob
import torch
from urllib.request import urlretrieve
from PIL import Image
from IPython.display import Image, display
import streamlit as st
from fastai import *
from fastai.vision.widgets import *
from fastai.vision.all import *

url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)




class Predict:
    def __init__(self, filename):
        self.learn_inference = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= filename)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
 
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])

        if uploaded_file is not None:
            return PILImage.create((uploaded_file))
        return None
        
        
        

    def display_output(self):
            self.learn_inference.results =  self.learn_inference(self.img, size=160)  # includes NMS
            self.learn_inference.results.print()  
            self.learn_inference.results.save()
            st.image('/results/*.jpg')
            
if __name__=='__main__':
    predictor = Predict(filename)
    
    
