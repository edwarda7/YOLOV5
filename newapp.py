import glob
import torch
from urllib.request import urlretrieve

url = ("http://dl.dropboxusercontent.com/s/fkdy4rbf8g8wm2s/best.pt?raw=1")
filename = "best.pt"
urlretrieve(url,filename)

model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=filename)

from PIL import Image

im_paths = glob.glob('/content/test/images/*.jpg')

for i in range(len(im_paths)):
  img = Image.open(im_paths[i])
  results = model(img, size=160)  # includes NMS
  results.print()  
  results.save()
  
