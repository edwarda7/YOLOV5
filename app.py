import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image
import glob


from PIL import Image
from IPython.display import Image, display



x = "best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
model = model.autoshape()
uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])




class Predict:
    def __init__(self, x):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
        self.display_output()
     
            
            
  

    def display_output(self):
        for i in range(len(im_paths)):
            img = Image.open(im_paths[i])
            results = model(img, size=160)  # includes NMS
            results.print()  
            results.save()

if __name__=='__main__': 
    predictor = Predict(x)




