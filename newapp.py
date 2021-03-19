import glob
import torch
from urllib.request import urlretrieve
from PIL import Image


model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)



im_paths = glob.glob('/content/test/images/*.jpg')

for i in range(len(im_paths)):
  img = Image.open(im_paths[i])
  results = model(img, size=160)  # includes NMS
  results.print()  
  results.save()
  
  
from IPython.display import Image, display

for imageName in glob.glob('uploaded_file'): #assuming JPG
    st.image(Image(filename=imageName))
