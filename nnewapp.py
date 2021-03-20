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




class Predict:
    def __init__(self, filename):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
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
        for i in range(len(im_paths)):
            img = Image.open(im_paths[i])
            results = model(img, size=160)  # includes NMS
            results.print()  
            results.save()
        for imageName in glob.glob('/results/*.jpg'): 
            display(Image(filename=imageName))
            st.image(Image(filename=imageName))
            
if __name__=='__main__':
    predictor = Predict(x)
    
    
    
    
st.image(Image(filename=imageName))


