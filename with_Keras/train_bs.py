# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
import keras
from keras import applications
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers.core import Flatten,Reshape
from imutils import paths
from net.vgg19 import VGG19
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import sys
import json
from lenet import LeNet
from my_model import MyModel

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
set_session = tf.Session(config=config)

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 1000
INIT_LR = 0.001
BS = 32
CLASS_NUM = 5031
norm_size = 64

def generate_arrays_from_json(path, batch_size):
    with open(path,'r') as f:
        dataset = json.load(f)
    while 1:
        ind = 0
        X = []
        Y = []
        for i in dataset:
            x = i['image']
            y = i['label']
            x = cv2.imread(x, cv2.IMREAD_COLOR)
            x = cv2.resize(x, (norm_size,norm_size))
            x = img_to_array(x)
            y = np.array(y)
            X.append(x)
            Y.append(y)
            ind += 1
            if ind == batch_size:
                ind = 0
                X = np.array(X, dtype='float') / 255.0
                Y = to_categorical(Y, num_classes=CLASS_NUM)
                yield(np.array(X),np.array(Y))
                X=[]
                Y=[]

def train(aug,train_path, test_path, l_train, l_test):
    # initialize the model
    print("[INFO] compiling model...")
    base_model = VGG19(weights='imagenet', include_top=False, classes=CLASS_NUM, input_shape=(64,64,3))
    output = base_model.output
    output = GlobalAveragePooling2D()(output)
    print(output.shape)
    #output = Flatten()(output)
    #output = Reshape((8192,))(output)
    
    print(output.shape)
    output = Dense(2048, activation='relu')(output)
    predictions = Dense(5031, activation='softmax')(output)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainble = False

    #model = LeNet.build(width=norm_size, height=norm_size, depth=1, classes=CLASS_NUM)
    #model = MyModel.build(width=norm_size, height=norm_size, depth=1, classes=CLASS_NUM)
    #model = InceptionV3(weights='imagenet', include_top=True)
    #model = InceptionV3.build(width=norm_size, height=norm_size, depth=1, calsses=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(generate_arrays_from_json(train_path, batch_size=BS),
        validation_data=generate_arrays_from_json(test_path, batch_size=BS), steps_per_epoch=l_train // BS,
        validation_steps=l_test//BS,
        epochs=EPOCHS, verbose=1, callbacks=[TensorBoard(log_dir='mytensorboard/vgg16')])

    # save the model to disk
    print("[INFO] serializing network...")
    model.save('./model')
    


#python train.py --dataset_train ../../traffic-sign/train --dataset_test ../../traffic-sign/test --model traffic_sign.model
if __name__=='__main__':
    train_file_path = 'train.json'
    test_file_path = 'test.json'
    print('Start training')
    with open(train_file_path,'r') as f:
        tem = json.load(f)
        #print(tem)
        l_train = len(tem)
    with open(test_file_path,'r') as f:
        tem = json.load(f)
        l_test = len(tem)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    train(aug,train_file_path, test_file_path, l_train, l_test)
