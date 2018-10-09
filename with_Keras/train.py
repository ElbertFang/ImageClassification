# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
from lenet import LeNet
from my_model import MyModel

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
set_session = tf.Session(config=config)

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 1000
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 5031
norm_size = 64


def load_data(path):
    name_list = os.listdir(path)
    data = []
    labels = []
    imagePaths = []
    for i in name_list:
        path_dir = path + i + '/'
        for j in os.listdir(path_dir):
            path_now = path_dir + j
            imagePaths.append(path_now)
            label = int(i)
            labels.append(label)
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)                         
    return data,labels
    

def train(aug,trainX,trainY,testX,testY):
    # initialize the model
    print("[INFO] compiling model...")
    model = LeNet.build(width=norm_size, height=norm_size, depth=1, classes=CLASS_NUM)
    #model = MyModel.build(width=norm_size, height=norm_size, depth=1, classes=CLASS_NUM)
    #model = InceptionV3(weights='imagenet', include_top=True)
    #model = InceptionV3.build(width=norm_size, height=norm_size, depth=1, calsses=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save('./model')
    


#python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__=='__main__':
    train_file_path = '/home/public_datasets/dzj_img_5k/train/'
    test_file_path = '/home/public_datasets/dzj_img_5k/test/'
    trainX,trainY = load_data(train_file_path)
    testX,testY = load_data(test_file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    train(aug,trainX,trainY,testX,testY)