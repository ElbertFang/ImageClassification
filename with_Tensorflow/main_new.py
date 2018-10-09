import os
import cv2
import time
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
tf.app.flags.DEFINE_integer('test_batchsize', 20, 'the number of pair image&label in a test batch')
tf.app.flags.DEFINE_integer('num_classes', 200, 'the number of classes in the dataset')

nBatchs = int(16000 * FLAGS.num_epochs / FLAGS.batch_size)
test_nBatchs = int(4000 / FLAGS.batch_size)
#训练集和测试集使用batch后的最大迭代次数
max_steps = int(16000 * FLAGS.num_epochs / FLAGS.batch_size)
test_max_steps = int(4000/FLAGS.test_batchsize)

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

def get_dataset(file_name):
    dataset = tf.data.TFRecordDataset(file_name)
    dataset = dataset.map(map_data)
    return dataset

def main():
    #检查是否已有tfrecords文件，如果没有则重新提取
    if os.path.isfile(FLAGS.train_path) and os.path.isfile(FLAGS.test_path):
        print('The tfrecords of train and test havd been token before.')
    else:
        print('Can not find the tfrecords files.')
        print('Make tfrecords now.')
        take_tfrecords()
        print('Tfrecords files have been takon.')

    #建立训练数据集
    train_dataset = get_dataset(FLAGS.train_path)
    train_dataset = train_dataset.shuffle(10000).repeat().batch(FLAGS.batch_size)
    #生成迭代器
    train_iterator = train_dataset.make_one_shot_iterator()
    next_elements_tr = train_iterator.get_next()

    #建立测试数据集
    test_dataset = get_dataset(FLAGS.test_path)
    test_dataset = test_dataset.shuffle(10000).repeat().batch(FLAGS.test_batchsize)
    #生成迭代器
    test_iterator = test_dataset.make_one_shot_iterator()
    next_elements_te = test_iterator.get_next()

    x = tf.placeholder(tf.float32, [None, 64, 64, 1])
    y_= tf.placeholder(tf.int32, [None])
    y_oh = tf.one_hot(indices=y_, depth=FLAGS.num_classes)
    #keep_prob：dropout时保存的概率
    keep_prob = tf.placeholder(tf.float32)
    #使用模型进行测试
    y = my_model(x, keep_prob)
    #loss与优化器的选择
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_oh, logits = y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    #判断预测是否正确
    correct_prediction = tf.equal(tf.argmax(y_oh,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    train_pre_now = 0
    for step in range(max_steps):
        start_time = time.time()
        image_batch_tr, label_batch_tr = sess.run(fetches=next_elements_tr)
        _, train_loss, _, train_pre = sess.run([optimizer, loss, accuracy, correct_prediction], feed_dict={x: image_batch_tr, y_: label_batch_tr, keep_prob: 1.0})
        duration = time.time() - start_time
        train_pre_now += np.sum(train_pre)

        if step % 500 == 0:
            train_acc = train_pre_now/(500*FLAGS.batch_size)
            train_pre_now = 0
            test_pre_now  = 0
            print('Training loss is %.5f and accuracy is %.5f'%(train_loss, train_acc))
            for _ in range (test_max_steps):
                image_batch_te, label_batch_te = sess.run(fetches=next_elements_te)
                test_loss, test_pre = sess.run([loss, correct_prediction], feed_dict={x: image_batch_te, y_: label_batch_te, keep_prob: 1.0})
                test_pre_now += np.sum(test_pre)
            prediction = test_pre_now/4000
            print('Current loss of test split is %.5f and accuracy is %.5f.'%(test_loss, prediction))

if __name__ == '__main__':
    main()
