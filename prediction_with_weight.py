#!/usr/bin/python

import sys
import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from helpers import load_all,save_image,load_image
import numpy as np

batch_size = 32
img_size = 400
test_img_size = 608
window_size = 72
patch_size = 16
pred_dir = './predictions/'
mode = 'rgb'

def prediction_to_img(pred, img_size, patch_size,threshold):
    small_dim = img_size // patch_size
    small_img = (pred > threshold).astype(int).reshape((small_dim, small_dim))
    return np.repeat(np.repeat(small_img, patch_size, axis=0), patch_size, axis=1)

def extend_img(img, size):
    '''extend the size of the image by reflecting at the borders so the window always fit in'''
    if (len(img.shape) == 3):
        r = np.pad(img[:, :, 0], size, 'reflect')
        g = np.pad(img[:, :, 1], size, 'reflect')
        b = np.pad(img[:, :, 2], size, 'reflect')
        return np.stack([r, g, b], axis=2)
    else:
        return np.pad(img, size, 'reflect')

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

def main():
    # print command line arguments
    if(len(sys.argv) < 2):
        print("not enought arguments, put the weight file as argument")
        return
    weight_path = sys.argv[1]

    reg = 1e-6

    model = Sequential()

    model.add(Conv2D(32, (3, 3),  padding = 'same' ,input_shape=(window_size, window_size, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512 , kernel_regularizer=l2(reg)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(reg)))

    opt = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.2, patience=5, min_lr=0.0)

    model.load_weights(weight_path)

    nb_pred = 50
    pred_path = './test_set_images/test_'
    save_path = pred_dir

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    for i in range(1, nb_pred+1):
        to_predict_img = load_image(pred_path+ str(i)+'/test_'+str(i)+'.png', mode)
        to_predict_windows = windows_from_img(to_predict_img, window_size, patch_size)
        to_predict_windows = np.asarray(to_predict_windows)
        pred = model.predict(to_predict_windows, batch_size)
        save_image(save_path + 'pred_' + str(i) + '.png', prediction_to_img(pred, test_img_size, patch_size, threshold=0.4))

if __name__ == "__main__":
    main()
