import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) # for reproducibility

# Mnist_Data load
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# layer & variable -> wide, deep hypothesis definition
# initializer=tf.contrib.layers.xavier_initializer() --> tf.random_normal을 써서 weight를 초기화하는 것이 아닌 xavier 함수를 사용해서 weight를 초기화함.
# xavier : input 개수, output 개수 w = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)
# 초기부터 cost값이 작게 나온다는 것은 weight 초기 값을 잘 잡아주었다는 것이다.
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.get_variable("w1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
layer1 = tf.nn.relu(tf.matmul(X, W1)+b1)

W2 = tf.get_variable("w2", shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2)+b2)

W3 = tf.get_variable("w3", shape=[256, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([128]))
layer3 = tf.nn.relu(tf.matmul(layer2, W3)+b3)

W4 = tf.get_variable("w4", shape=[128, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(layer3, W4)+b4

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
            cost_val, train_val = sess.run([cost, train], feed_dict={X:batch_x, Y: batch_y})
            cost_avg += cost_val/iterations
        print(f"epoch : {epoch}, cost : {cost_avg}")
    print("learning finish")

    # Accuracy
    p, a = sess.run([predict, accuracy], feed_dict={X: mnist.test.images, Y:mnist.test.labels})
    print(f"accuracy = {a}")

    # Predict 1 data Verification
    r = random.randint(0, mnist.test.num_examples - 1)
    print(f"label = {sess.run(tf.argmax(mnist.test.labels[r : r+1],1))}")
    print(f"prediction = {sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r:r+1]})}")

    # plt.show(
    #     mnist.test.images[r:r+1].reshape(28, 28),
    #     cmap='Greys',
    #     interpolation = 'nearest'
    # )
    # plt.show()