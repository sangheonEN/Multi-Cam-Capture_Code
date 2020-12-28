import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777)  # for reproducibility

X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W1 = tf.Variable(tf.random_normal([2, 2]), name="weight1")
b1 = tf.Variable(tf.random_normal([2]), name="bias1")

W2 = tf.Variable(tf.random_normal([2, 1]), name="weight2")
b2 = tf.Variable(tf.random_normal([1]), name="bias2")

# layer 추가
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)
# layer를 입력데이터로 최종 Weight에 넣음
logits = tf.matmul(layer1, W2)+b2
hypothesis = tf.sigmoid(logits)

cost = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        train_val = sess.run([cost, train], feed_dict={X: X_data, Y: Y_data})
        if step % 100 == 0:
            print(f"step = {step}, cost = {sess.run(cost, feed_dict={X: X_data, Y: Y_data})}, \nW1 = {sess.run(W1)}, W2 = {sess.run(W2)}")
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: X_data, Y: Y_data})
    print(f"predicted = {p}, accuracy = {a}, H(x) = {h}")


