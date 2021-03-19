import glob
import torch
from urllib.request import urlretrieve
from PIL import Image
from IPython.display import Image, display
import streamlit as st
from fastai import *
from fastai.vision.widgets import *
from fastai.vision.all import *


model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model="best.pt")




uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
im_paths = glob.glob('uploaded_file')


for i in range(len(im_paths)):
  img = Image.open(im_paths[i])
  results = model(img, size=160)  # includes NMS
  results.print()  
  results.save()
  
  

for imageName in glob.glob('uploaded_file'): #assuming JPG
    st.image(display(Image(filename=imageName)))
