from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
from utilities import *
from PIL import ImageFont, ImageDraw, ImageEnhance

n=1920
m=1080

pil_im = Image.new('RGB', (n, m))
np_im = np.asarray(pil_im)

draw = ImageDraw.Draw(pil_im)

print(np_im.shape)

for bbox in make_crop_coordinates_nhz(np_im,"9_16"):    
    draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])),  outline="white")


