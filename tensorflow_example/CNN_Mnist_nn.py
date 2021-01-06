import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) # for reproducibility

# Mnist_Data load
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# CNN START
sess = tf.InteractiveSession()

# hyper parameters
learning_rate = 0.001
train_epoch = 15
batch_size = 100

# X_data, Y_data, Weight, layer definition
X = tf.placeholder(tf.float32, shape=[None, 784])
print(f"X = {X}")
Y = tf.placeholder(tf.float32, shape=[None, 10])
X_img = tf.reshape(X, [-1, 28, 28, 1])

# filter (weight) definition
# layer1
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding="SAME")
print(f"conv1 shape = {L1}")
L1 = tf.nn.relu(L1)
print(f"relu1 shape = {L1}")
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
print(f"max_pool1 shape = {L1}")

# layer2
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding="SAME")
print(f"conv2 shape = {L2}")
L2 = tf.nn.relu(L2)
print(f"relu2 shape = {L2}")
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
print(f"max_pool2 shape = {L2}")

# 1줄로 펼치기 위해서 reshape을 사용한다~ n개의 데이터니 -1, filter를 거쳐서 최종 데이터의 수
L2_flat = tf.reshape(L2, [-1, 7*7*64])
print(f"FULLY CONNECTED LAYER에 넣기 전 데이터 전처리 shape = {L2}")
print(f"결과적으로 X의 데이터 784개가 CONV2개와 POOLING을 LAYER를 거치면서 3136개가 되었다.")

# Final weight
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]), name="bias")

# hypothesis Definition
hypothesis = tf.matmul(L2_flat, W3)+b

# cost Function Definition
cost = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y)

# optimizer Definition
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# sess
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epoch):
        cost_avg = 0
        iteration = int(mnist.train.num_examples / batch_size)
        for i in range(iteration):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs, Y:batch_ys}
            cost_val, train_val = sess.run([cost, train], feed_dict=feed_dict)
            cost_avg += cost_val / iteration
        print(f"epoch = {epoch+1}, cost = {cost_avg}")
    print("Train Finish")

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(f"Accuracy = {sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})}")

    # Prediction
    r = random.randint(0, mnist.test.num_examples - 1)
    print(f"label = {sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))}")
    print(f"prediction = {sess.run(tf.argmax(hypothesis, 1),feed_dict={X:mnist.test.images[r:r+1]})}")

    plt.imshow(mnist.test.images[r:r + 1].
              reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()



