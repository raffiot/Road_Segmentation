
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

#Images are treated in hsv if not explicitly specified
def load_image(infilename, mode='hsv'):
    data = mpimg.imread(infilename)
    if len(data.shape) == 3 and mode is 'hsv':
        data = rgb_to_hsv(data)
    return data

#The image are saved in black and white
def save_image(outfilename, data):
    data3 = Image.fromarray((data * 255).astype(np.uint8))
    data3.save(outfilename)

def load_all(path,mode='hsv'):
    files = os.listdir(path)
    n = len(files)
    print('loading '+str(n)+' images')
    imgs = [load_image(path + files[i],mode) for i in range(n)]
    return imgs,n
