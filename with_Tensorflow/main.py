import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"]='3'

from take_tfrecords import take_tfrecords
from my_model import my_model
#from nets import nets_factory
FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('file_train', '/home/public_datasets/dzj/data_200c/train/', 'path of train image data')
tf.app.flags.DEFINE_string('file_test', '/home/public_datasets/dzj/data_200c/test/', 'path of train image data')
tf.app.flags.DEFINE_string('train_path', 'train.tfrecords', 'path of train data')
tf.app.flags.DEFINE_string('test_path', 'test.tfrecords', 'path of test data')

tf.app.flags.DEFINE_integer('num_epochs', 1000, 'the number of training epochs')
tf.app.flags.DEFINE_integer('image_size', 64, 'the length and width of images')
tf.app.flags.DEFINE_integer('batch_size', 32, 'the number of pair image&label in a batch')
tf.app.flags.DEFINE_integer('num_classes', 200, 'the number of classes in the dataset')

nBatchs = int(16000 * FLAGS.num_epochs / FLAGS.batch_size)
test_nBatchs = int(4000 / FLAGS.batch_size)

def map_data(example_data):
    features = (tf.parse_single_example(example_data, 
        features = {
            'image' : tf.FixedLenFeature([], tf.string),
            'label' : tf.FixedLenFeature([], tf.int64)
        }))
    image = tf.decode_raw(features['image'], tf.float32)
    #image = tf.image.decode_jpeg(features['image'], tf.float32)
    image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, 1])
    label = tf.cast(features['label'], tf.int32)
    return image, label

def dataset(tf_file):
    split_dataset = tf.data.TFRecordDataset(tf_file)
    split_dataset = split_dataset.map(map_data)
    return split_dataset

def main():
    #检查是否已有tfrecords文件，如果没有则重新提取
    if os.path.isfile(FLAGS.train_path) and os.path.isfile(FLAGS.test_path):
        print('The tfrecords of train and test havd been token before.')
    else:
        print('Can not find the tfrecords files.')
        print('Make tfrecords now.')
        take_tfrecords()
        print('Tfrecords files have been takon.')

    train_data = dataset(FLAGS.train_path)
    train_data = train_data.shuffle(10000).batch(FLAGS.batch_size).repeat(FLAGS.num_epochs)
    #train_iterator = train_data.make_initializable_iterator()
    train_iterator = train_data.make_initializable_iterator()

    test_data = dataset(FLAGS.test_path)
    test_data = test_data.batch(8000)
    test_iterator = test_data.make_initializable_iterator()
    # handle = tf.placeholder(tf.string, shape=[])
    # iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
    
    # x, y_ = iterator.get_next()

    train_x, train_y_ = train_iterator.get_next()
    test_x, test_y_ = test_iterator.get_next()
    x = tf.placeholder(tf.float32, [None, 64, 64, 1])
    y_= tf.placeholder(tf.int32, [None])
    #y_ = tf.one_hot(y_, depth = FLAGS.batch_size)
    y_oh = tf.one_hot(indices=y_, depth=FLAGS.num_classes)
    keep_prob = tf.placeholder(tf.float32)
    y = my_model(x, keep_prob)

    #x = tf.Print(x,[x])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_oh, logits = y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_oh,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)
        # handle_train = sess.run(train_iterator.string_handle())
        #train_image, train_label = sess.run([train_x, train_y_])
        #test_image, test_label = sess.run([test_x, test_y_])
        for i in range(nBatchs):
            while True:
                try:
                    train_image, train_label = sess.run([train_x, train_y_])
                    i+=1
                    #_, acc, curr_loss = sess.run([optimizer, accuracy, loss], feed_dict={handle: handle_train, keep_prob:0.5})
                    #aaa, bbb = sess.run([x,y_], feed_dict={x:image_batch, y_:label_batch})
                    #print(sess.run([tf.shape(aaa), tf.shape(bbb)]))            
                    #ccc = sess.run(y, feed_dict={x:image_batch, y_:label_batch, keep_prob:0.5})
                    #print(sess.run(tf.shape(ccc)))
                    _, train_acc, train_loss = sess.run([optimizer, accuracy, loss], feed_dict={x: train_image, y_: train_label, keep_prob:1.0})
                    #sess.run(optimizer, feed_dict={x: image_batch, y_: label_batch, keep_prob:0.5})
                    if i%(16000/32) == 0:
                        while True:
                            try:
                                test_image, test_label = sess.run([test_x, test_y_])
                                test_loss, test_acc = sess.run([loss, accuracy], feed_dict={x: test_image, y_:test_label, keep_prob:1.0})
                            except tf.errors.OutOfRangeError:
                                break
                        print('Train loss %.5f accuracy %.5f' % (train_loss, train_acc))
                        print('[%d of %d]Current loss is %.5f Current accuracy is %.5f' % (i, nBatchs,test_loss, test_acc))
                except tf.errors.OutOfRangeError:
                                break

if __name__ == '__main__':
    main()
