import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from helpers import load_all,save_image,load_image

def main():
    mode = 'rgb'
    root_dir = "./training/"

    image_dir = root_dir + "images/"
    gt_dir = root_dir + "groundtruth/"

    imgs,_ = load_all(image_dir,mode)
    gt_imgs,_ = load_all(gt_dir,mode)
    count = 0

    #If augmented_training does not exist we create all directories.
    new_image_dir = './augmented_training/satellite/'
    new_gt_dir = './augmented_training/ground_truth/'

    if not os.path.exists('./augmented_training/'):
            os.makedirs('./augmented_training/')
            os.makedirs(new_image_dir)
            os.makedirs(new_gt_dir)



    for i, (img, gt_img) in enumerate(zip(imgs, gt_imgs)):
        tmp = img
        gt_tmp = gt_img
        for k in range(2):
            for j in range(4):
                save_image(new_image_dir + 'sat_{}.png'.format(count), tmp)
                save_image(new_gt_dir + 'gt_{}.png'.format(count), gt_tmp)
                tmp = np.rot90(tmp)
                gt_tmp = np.rot90(gt_tmp)
                count += 1
            tmp = np.flip(tmp, 0)
            gt_tmp = np.flip(gt_tmp, 0)

    #We produce 8 images from one image (included)

if __name__ == "__main__":
    main()
