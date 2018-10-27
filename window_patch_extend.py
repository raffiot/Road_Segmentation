import numpy as np
import os,sys

def extend_img(img, size):
    '''extend the size of the image by reflecting at the borders so the window always fit in'''
    if (len(img.shape) == 3):
        r = np.pad(img[:, :, 0], size, 'reflect')
        g = np.pad(img[:, :, 1], size, 'reflect')
        b = np.pad(img[:, :, 2], size, 'reflect')
        return np.stack([r, g, b], axis=2)
    else:
        return np.pad(img, size, 'reflect')

#-----------With these methods we cut out our image in window that will be the input data of our neural net.------

def patch_indices(image_size,patch_size):
    '''cut out the image into patch and return the indices of the center of each patch'''
    indices = []
    for i in range(patch_size // 2,image_size,patch_size):
        for j in range(patch_size // 2,image_size,patch_size):
            indices.append(np.asarray([i,j]))

    return indices

def get_window(index, image, window_size):
    '''Return a window around the pixel at index'''
    #assume window always fit into image
    #window_size have to be odd
    size = window_size // 2
    x1 = index[0] - size
    x2 = index[0] + size
    y1 = index[1] - size
    y2 = index[1] + size
    return image[x1 : x2, y1 : y2]

def windows_from_img(img, window_size, patch_size):
    '''return a list of window that correspond to each patch'''
    indices = patch_indices(img.shape[0], patch_size)
    ext_img = extend_img(img, window_size // 2)
    windows = []
    for index in indices:
        windows.append(get_window(index + window_size // 2, ext_img, window_size))

    return windows

#--------These methods are used to generate the labels (output of neural net) of each patch of one training ground truth image-----------

def patch_mean(Y, index, patch_size):
    '''Compute and return the mean value of a patch'''
    i = index[0]
    j = index[1]
    x_start = i - patch_size // 2
    x_end   = i + patch_size // 2
    y_start = j - patch_size // 2
    y_end   = j + patch_size // 2

    Y_patch = Y[x_start : x_end, y_start : y_end]

    return np.mean(Y_patch)

def road_classification(Y, index, patch_size, threshold=0.5, verbose=False):
    '''Returned as (road, background)'''
    m = patch_mean(Y, index, patch_size)

    if verbose:
        if m > threshold:
            print('road')
        else:
            print('background')

    return int(m > threshold)

def labels_from_gt_img(gt_img, window_size, patch_size):
    '''return a list of labels (road,background) for each patch of the ground_truth image'''
    #We take the indices of patches before we extend the image !
    indices = patch_indices(gt_img.shape[0], patch_size)
    ext_gt_img = extend_img(gt_img, window_size // 2)
    labels = []
    for index in indices:
        labels.append(road_classification(ext_gt_img, index + window_size // 2, patch_size))

    return labels

#Method to build a prediction image of img_size from an array of 0s and 1s corresponding to output label for each patch
def prediction_to_img(pred, img_size, patch_size, threshold=0.5):
    small_dim = img_size // patch_size
    small_img = (pred > threshold).astype(int).reshape((small_dim, small_dim))
    return np.repeat(np.repeat(small_img, patch_size, axis=0), patch_size, axis=1)
