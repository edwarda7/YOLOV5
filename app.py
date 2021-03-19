import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model='best.pt')
        
            

uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])
     
PILImage.create((uploaded_file))



st.image(self.img.to_thumb(500,500), caption='Uploaded Image')
 
result = model(uploaded_file, size=640)  # includes NMS
result.print()  
result.save() 
st.image("results/uploaded_file")


   




