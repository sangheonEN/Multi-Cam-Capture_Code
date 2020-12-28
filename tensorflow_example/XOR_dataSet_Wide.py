import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W1 = tf.Variable(tf.random_normal([2, 10]))
b1 = tf.Variable(tf.random_normal([10]))
layer = tf.sigmoid(tf.matmul(X, W1)+ b1)
W2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(layer, W2)+b2)

cost = tf.reduce_mean(-Y*tf.log(hypothesis)-(1-Y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, train_val = sess.run([cost, train], feed_dict={X:X_data, Y: Y_data})
        if step % 1000 == 0:
            print(f"cost = {cost_val}")
    p, a, h = sess.run([predicted, accuracy, hypothesis], feed_dict={X: X_data, Y: Y_data})
    print(f"predicted = {p}, accuracy = {a}, H(x) = {h}")




