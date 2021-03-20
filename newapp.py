import glob
import torch
from urllib.request import urlretrieve
from PIL import Image
from IPython.display import Image, display
import streamlit as st
from fastai import *
from fastai.vision.widgets import *
from fastai.vision.all import *



x = "best.pt"

im_paths = glob.glob('uploaded_file')


model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)


class Predict:
    def __init__(self, filename):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()
 
            
    
    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
        if uploaded_file is not None:
            img =  Image.open("BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg")
        return None
        
        
        

    def display_output(self):
     
        for imageName in glob.glob('/results/*.jpg'): 
            st.image(Image(imageName))
            

    def get_prediction(self):
        if st.button('Classify'):
            results = model(self.img, size=160)  # includes NMS
            results.print()  
            results.save()
            st.write(results.print())
        else: 
            st.write(f'Click the button to classify') 

if __name__=='__main__':
    predictor = Predict(x)
    
