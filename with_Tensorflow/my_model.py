import tensorflow as tf

def my_model(x, keep_prob):
    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable('weight', [5,5,1,32], initializer= tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(x, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('conv2'):
        conv2_weights = tf.get_variable('weight', [5,5,32,64], initializer= tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('conv3'):
        conv3_weights = tf.get_variable('weight', [3,3,64,128], initializer= tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1,1,1,1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        pool3 = tf.nn.max_pool(relu3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    
    with tf.variable_scope('conv4'):
        conv4_weights = tf.get_variable('weight', [3,3,128,256], initializer= tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1,1,1,1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        pool4 = tf.nn.max_pool(relu4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        pool_shape = pool4.get_shape().as_list()
        length = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool4, [-1, length])

    with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable('weight',[length, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_bias    = tf.get_variable('bias',[1024], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_bias)
        #fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable('weight',[1024, 200], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias    = tf.get_variable('bias',[200], initializer=tf.constant_initializer(0.1))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_bias

    return fc2
