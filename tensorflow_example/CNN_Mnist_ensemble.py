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
batch_size = 100

# Model Class definition
class Model:
    # __init__ 을 사용하면 이 메서드는 생성자가 됨
    # 생성자 : 객체가 생성될 때 자동으로 호출되는 메서드를 의미
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # drop-out keep_porb
            self.keep_porb = tf.placeholder(tf.float32)

            # Input data placeholder
            self.X = tf.placeholder(tf.float32, shape=[None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, shape=[None, 10])

            # 32 개의 conv를 만들기 위해 filter의 갯수를 32개로 하고, Gray scale.
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding="SAME")
            print(f"L1 conv shape = {L1.shape}")
            L1 = tf.nn.relu(L1)
            L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")
            print(f"L1 max_pool shape = {L1.shape}")
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_porb)

            # L2 INPUT LAYER [?, 14, 14 32]
            W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding="SAME")
            print(f"L2 conv shape = {L1.shape}")
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            print(f"L2 max_pool shape = {L1.shape}")
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_porb)
            L2_Flat = tf.reshape(L2, [-1, 7*7*64])

            # L3 FIRST FULLY CONNECTED 7*7*64
            W3 = tf.get_variable("W3", shape=[7*7*64, 3136], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([3136]), name="bias1")
            L3 = tf.nn.relu(tf.matmul(L2_Flat, W3)+b3)
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_porb)

            # L4 SECOND FULLY CONNECTED INPUTS 3136 -> OUTPUTS 1024
            W4 = tf.get_variable("W4", shape=[3136, 1024], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([1024]), name="bias_s")
            L4 = tf.nn.relu(tf.matmul(L3, W4)+b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_porb)

            # L5 FINAL FULLY CONNECTED INPUTS 1024 -> OUTPUTS 10
            W5 = tf.get_variable("W5", shape=[1024, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]), name="bias_f")
            self.logits = tf.matmul(L4, W5)+b5

        # cost function definition
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train = self.optimizer.minimize(self.cost)

        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def prediction(self, x_test, keep_prob=1.0):
        return self.sess.run(self.logits, feed_dict={self.X: x_test, self.keep_porb:keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.X:x_test, self.Y:y_test, self.keep_porb:keep_prob})

    def train_start(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.train], feed_dict={self.X: x_data, self.Y: y_data, self.keep_porb:keep_prob})

# model train start
with tf.Session() as sess:
    sess = tf.Session()
    m1 = Model(sess, "M1")
    sess.run(tf.global_variables_initializer())
    print("train start")
    for epoch in range(train_epoch):
        cost_avg = 0
        iterlation = int(mnist.train.num_examples / batch_size)
        for i in range(iterlation):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost_val, train_val = m1.train_start(batch_x, batch_y)
            cost_avg += cost_val / iterlation
        print(f"epoch : {epoch}, cost : {cost_avg}")
    print("Learning Finish")

    # Accuracy result
    print(f"Accuracy : {m1.get_accuracy(mnist.test.images, mnist.test.labels)}")




    #
    # def predict(self, x_test, keep_prob = 1.0):
    #     return self.sess.run(self.logits, feed_dict=)
