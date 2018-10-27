import numpy as np
import os,sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from helpers import *
from window_patch_extend import *

mode = 'rgb'
batch_size = 128
epochs = 1
img_size = 400
test_img_size = 608
window_size = 72
patch_size = 16

#If you want to modify training and validation set size, please change numbers in ranges

def main():
    K.tensorflow_backend._get_available_gpus()


    #Load augmented images
    root_dir = "./augmented_training/"

    image_dir = root_dir + "satellite/"
    gt_dir = root_dir + "ground_truth/"


    imgs,nb_imgs = load_all(image_dir,mode)
    gt_imgs,nb_gt_imgs = load_all(gt_dir,mode)


    #shuffle images and ground truth before generating windows and patches
    imgs = np.asarray(imgs)
    gt_imgs = np.asarray(gt_imgs)

    per = np.random.permutation(imgs.shape[0])
    imgs = imgs[per]
    gt_imgs = gt_imgs[per]

    #Generate training set
    windows = []
    labels = []
    #We take as training set images from 0 to 200
    #If you take more becarefull with memory overflow
    for i in range(0, 100):
        windows += windows_from_img(imgs[i], window_size, patch_size)
        labels += labels_from_gt_img(gt_imgs[i], window_size, patch_size)

    windows = np.asarray(windows)
    labels = np.asarray(labels)

    per = np.random.permutation(windows.shape[0])
    windows = windows[per]
    labels = labels[per]

    #Generate test set
    windows_test = []
    labels_test = []

    for i in range(751, 800):
        windows_test += windows_from_img(imgs[i], window_size, patch_size)
        labels_test += labels_from_gt_img(gt_imgs[i], window_size, patch_size)

    windows_test = np.asarray(windows_test)
    labels_test = np.asarray(labels_test)

    per = np.random.permutation(windows_test.shape[0])
    windows_test = windows_test[per]
    labels_test = labels_test[per]

    #Saving weights option
    weigh_save_file = 'weights.hdf5'
    cp = keras.callbacks.ModelCheckpoint(weigh_save_file, monitor='acc', save_best_only=True, save_weights_only=True)

    reg = 1e-6

    #Building neural net
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
    model.summary()

    #Run training
    model.fit(windows, labels, batch_size=batch_size, epochs=epochs, callbacks=[cp, reduce_lr], validation_data=(windows_test, labels_test))
if __name__ == "__main__":
    main()
