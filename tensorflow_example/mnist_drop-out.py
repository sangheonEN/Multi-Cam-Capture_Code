import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) # for reproducibility

# Mnist_Data load
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# layer & variable -> wide, deep hypothesis definition
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

# drop_out rate 0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("w1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
layer1 = tf.nn.relu(tf.matmul(X, W1)+b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.get_variable("w2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2)+b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("w3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
layer3 = tf.nn.relu(tf.matmul(layer2, W3)+b3)
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

W4 = tf.get_variable("w4", shape=[512, 128], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([128]))
layer4 = tf.nn.relu(tf.matmul(layer3, W4)+b4)
layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

W5 = tf.get_variable("w5", shape=[128, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(layer4, W5)+b5

# cost Function definition
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= hypothesis, labels=Y))

# optimizer definition
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

# Predicted, Accuracy definition
predict = tf.argmax(hypothesis, 1)
correct = tf.equal(tf.argmax(Y, 1), predict)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# epoch, batch size
epoch_num = 20
batch_size = 100
iterations = int(mnist.train.num_examples / batch_size)

# Session Run
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        cost_avg = 0
        for i in range(iterations):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost_val, train_val = sess.run([cost, train], feed_dict={X:batch_x, Y: batch_y, keep_prob:0.7})
            cost_avg += cost_val/iterations
        print(f"epoch : {epoch}, cost : {cost_avg}")
    print("learning finish")

    # Accuracy
    p, a = sess.run([predict, accuracy], feed_dict={X: mnist.test.images, Y:mnist.test.labels, keep_prob:1})
    print(f"accuracy = {a}")

    # Predict 1 data Verification
    r = random.randint(0, mnist.test.num_examples - 1)
    print(f"label = {sess.run(tf.argmax(mnist.test.labels[r : r+1],1))}")
    print(f"prediction = {sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1],keep_prob:1})}")

    # plt.show(
    #     mnist.test.images[r:r+1].reshape(28, 28),
    #     cmap='Greys',
    #     interpolation = 'nearest'
    # )
    # plt.show()