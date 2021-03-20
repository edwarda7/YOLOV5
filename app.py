import glob
import torch
from urllib.request import urlretrieve
from PIL import Image
import streamlit as st

url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)


uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])



im_paths = glob.glob('BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg')

for i in range(len(im_paths)):
  img = Image.open(im_paths[i])
  results = model(img, size=160)  # includes NMS
  results.print()  
  results.save()
from IPython.display import Image, display

for imageName in glob.glob('BloodImage_00002_jpg.rf.72d4182864da81e2fc804f5382965abc.jpg'): #assuming JPG
    display(Image(imageName))
    
