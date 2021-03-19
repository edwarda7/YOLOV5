import streamlit as st
from urllib.request import urlretrieve
import torch

from PIL import Image




x = "best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model= x)
model = model.autoshape()
uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg', 'jpg'])




class Predict:
    def __init__(self, filename):
        self.learn_inference = load_learner(filename)
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
        st.image(self.img.to_thumb(500,500), caption='Uploaded Image')
        pred, pred_idx, probs = self.learn_inference.predict(self.img)
        st.write(f'Prediction: {pred} red blood cell; Probability: {probs[pred_idx]:.04f}')

 

if __name__=='__main__':
    predictor = Predict(filename)
