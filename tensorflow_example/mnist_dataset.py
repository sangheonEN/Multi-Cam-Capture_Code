import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777) # for reproducibility

# 데이터 load
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 변수설정
num_class = 10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.int32, [None, num_class])
W = tf.Variable(tf.random_normal([784, num_class]), name="weight")
b = tf.Variable(tf.random_normal([num_class]), name="weight")

# 가설설정
logits = tf.matmul(X, W)+b
hypothesis = tf.nn.softmax(logits)

# COST FUNCTION 정의
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
cost = tf.reduce_mean(cost_i)

# optimizer 정의
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)

# prediction, accuracy 정의
prediction = tf.argmax(hypothesis, 1)
correct_train = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_train, tf.float32))

# session epoch 사용
epoch_num = 20
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoch_num):
        cost_avg = 0
        for i in range(num_iterations):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            train_val, cost_val = sess.run([train, cost], feed_dict={X:batch_x, Y: batch_y})
            cost_avg += cost_val / num_iterations
        print(f"epoch : {epoch}, cost : {cost_val}")
    print("Learning finished")

    # Accuracy
    print(f"accuracy : {sess.run(accuracy, feed_dict={X:mnist.test.images, Y: mnist.test.labels})}")











# # 이미지 데이터 픽셀 자료구조 확인
# (images_train, labels_train) = data_train
# (images_test, labels_test) = data_test
#
# images_train_f = np.float32(images_train)
# images_test_f = np.float32(images_train)
#
# print(images_train.shape, labels_train.shape)
# print(images_test.shape, labels_test.shape)
#
# a = [[1], [2], [3], [4] ,[5]]
#
# for i in a:
#     arr = []
#     arr.append(i)
#     print(arr)
#     print(type(arr))