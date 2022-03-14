import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) # for reproducibility

# Mnist_Data load
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
train_epoch = 15
train_batchsize = 100

# Model Class Definition
class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

    def _build_net(self):
        with tf.variable_scope(self.name):
            self.keep_porb = tf.placeholder(tf.float32)

