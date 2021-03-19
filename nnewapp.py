import glob
import torch
from urllib.request import urlretrieve
from PIL import Image
from IPython.display import Image, display
import streamlit as st




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
        
        
        

    def display_output(self):
        for i in range(len(im_paths)):
            img = Image.open(im_paths[i])
            results = model(img, size=160)  # includes NMS
            results.print()  
            results.save()

       
        for imageName in glob.glob('uploaded_file'): #assuming JPG
            display(Image(filename=imageName))
            st.image(imageName)
 

if __name__=='__main__':
    predictor = Predict(x)

