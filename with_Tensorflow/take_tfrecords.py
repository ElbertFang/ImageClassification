import os 
import cv2
import numpy as np 
import tensorflow as tf 

from PIL import Image

FLAGS = tf.app.flags.FLAGS

def take_tfrecords():
    train_list = []
    test_list  = []
    for k in ['train', 'test']:
        if k == 'train':
            path_split = FLAGS.file_train
            image_and_label = train_list
            save_path = FLAGS.train_path
        else:
            path_split = FLAGS.file_test
            image_and_label = test_list
            save_path = FLAGS.test_path
        for i in range(200):
            path_now = path_split + str(i) + '/'
            dir_list = os.listdir(path_now)
            file_list = [path_now + x for x in dir_list]
            for j in file_list:
                dic_now = {}
                dic_now['label'] = i
                dic_now['image'] = j
                image_and_label.append(dic_now)
        print('The image number of %s is %d' % (k,len(image_and_label)))

        writer = tf.python_io.TFRecordWriter(save_path)
        for i in image_and_label:
            image_path = i['image']
            label = i['label']
            # image = Image.open(image_path)
            # image = image.resize((FLAGS.image_size, FLAGS.image_size))
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (FLAGS.image_size, FLAGS.image_size))
            #image = image / 255.
            image = image.astype(np.float32)
            image_bytes = image.tobytes()

            example = tf.train.Example(features = tf.train.Features(
                feature = {
                    'image' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image_bytes])),
                    'label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
                }
            ))
            writer.write(example.SerializeToString())
        writer.close()

if __name__ == '__main__':
    take_tfrecords()