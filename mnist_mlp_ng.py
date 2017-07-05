# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 20:18:22 2017

@author: Niladri
# Snipet of the codes are taken from
# https://www.tensorflow.org/get_started/mnist/pros#weight_initialization
# for GPU : https://github.com/wookayin/tensorflow-talk-debugging/blob/master/codes/10-mnist.py
"""
#Load MNIST Data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Start TensorFlow InteractiveSession

import tensorflow as tf
sess = tf.InteractiveSession()

#Build a Softmax Regression Model

#Placeholder
# Start understanding code from 1st Convolutional Layer
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# Weight Initialization

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and Pooling

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # [stride in batch,x_stride,y_stride,stride_along_channel]
# padding is done is such a way so that input and output image size is same. For a filter of 5, padding is 2.
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME') # Same as convolution.


# First Convolutional Layer

W_conv1 = weight_variable([5, 5, 1, 32]) # Filter Size 5,5. Input Filter=1, Output Filter=2
b_conv1 = bias_variable([32])# There should bias term for each Output Filter

x_image = tf.reshape(x, [-1, 28, 28, 1])# Input Image Size is 28*28 and no of channels is 1. -1 is used to denote
# batch Size, so [batch Size,28,28,1] would be reshaped.

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# 2nd Convolutional Layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Dense Layer

W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 2 maxpooling layer becomes 28-->14-->7
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# DropOut

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Train and evaluate Model

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # learning rate 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#with tf.Session() as sess:
with tf.Session(config=tf.ConfigProto(
       gpu_options=tf.GPUOptions(allow_growth=True),
       device_count={'GPU': 1})) as sess: # For running in GPU
  sess.run(tf.global_variables_initializer()) # Initialization of variables
  for i in range(2000): # No of epochs
    batch = mnist.train.next_batch(100) # batch size 100
    if i % 100 == 0: # after every 100 epochs check how good your learnt model is--> Training accuracy
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0}) # during evaluation of model, al the nodes are active.
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # dropout 0.5 .During training 
    # only 50% nodes are active

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
#def main():
#    with tf.Session(config=tf.ConfigProto(
#        gpu_options=tf.GPUOptions(allow_growth=True),
#        device_count={'GPU': 1})) as session:
#        train(session)
#
#if __name__ == '__main__':
#    main()

