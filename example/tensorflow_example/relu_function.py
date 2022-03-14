import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.set_random_seed(777) # for reproducibility

X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.Variable(tf.random_uniform([2, 5], -1, 1))
W2 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
W3 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
W4 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
W5 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
W6 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
W7 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
W8 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
W9 = tf.Variable(tf.random_uniform([5, 1], -1, 1))

b1 = tf.Variable(tf.zeros([5]))
b2 = tf.Variable(tf.zeros([5]))
b3 = tf.Variable(tf.zeros([5]))
b4 = tf.Variable(tf.zeros([5]))
b5 = tf.Variable(tf.zeros([5]))
b6 = tf.Variable(tf.zeros([5]))
b7 = tf.Variable(tf.zeros([5]))
b8 = tf.Variable(tf.zeros([5]))
b9 = tf.Variable(tf.zeros([1]))


with tf.name_scope("layer1") as scope:
    layer1 = tf.nn.relu(tf.matmul(X, W1)+b1)
with tf.name_scope("layer2") as scope:
    layer2 = tf.nn.relu(tf.matmul(layer1, W2)+b2)
with tf.name_scope("layer3") as scope:
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
with tf.name_scope("layer4") as scope:
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
with tf.name_scope("layer5") as scope:
    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)
with tf.name_scope("layer6") as scope:
    layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)
with tf.name_scope("layer7") as scope:
    layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)
with tf.name_scope("layer8") as scope:
    layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)
with tf.name_scope("layer9") as scope:
    hypothesis = tf.sigmoid(tf.matmul(layer8, W9)+b9)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(-Y*tf.log(hypothesis)-(1-Y)*tf.log(1-hypothesis))
    tf.summary.scalar("cost", cost)

with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

with tf.name_scope("accuracy"):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    merge_summary = tf.summary.merge_all()
    write = tf.summary.FileWriter("./logs/aa")
    write.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, train_val, summary_val = sess.run([cost, train, merge_summary], feed_dict={X: X_data, Y:Y_data})
        write.add_summary(summary_val, global_step=step)
        if step % 1000 == 0:
            print(f"step = {step}, cost = {cost_val}")
    a, p = sess.run([accuracy, predicted], feed_dict={X:X_data, Y:Y_data})
    print(f"accuracy = {a}, predicted = {p}")



# sigmoid함수
# import tensorflow as tf
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# tf.set_random_seed(777) # for reproducibility
#
# X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Y_data = np.array([[0], [1], [1], [0]])
#
# X = tf.placeholder(tf.float32, shape=[None, 2])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
#
# W1 = tf.Variable(tf.random_uniform([2, 5], -1, 1))
# W2 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
# W3 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
# W4 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
# W5 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
# W6 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
# W7 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
# W8 = tf.Variable(tf.random_uniform([5, 5], -1, 1))
# W9 = tf.Variable(tf.random_uniform([5, 1], -1, 1))
#
# b1 = tf.Variable(tf.zeros([5]))
# b2 = tf.Variable(tf.zeros([5]))
# b3 = tf.Variable(tf.zeros([5]))
# b4 = tf.Variable(tf.zeros([5]))
# b5 = tf.Variable(tf.zeros([5]))
# b6 = tf.Variable(tf.zeros([5]))
# b7 = tf.Variable(tf.zeros([5]))
# b8 = tf.Variable(tf.zeros([5]))
# b9 = tf.Variable(tf.zeros([1]))
#
#
# with tf.name_scope("layer1") as scope:
#     layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)
# with tf.name_scope("layer2") as scope:
#     layer2 = tf.sigmoid(tf.matmul(layer1, W2)+b2)
# with tf.name_scope("layer3") as scope:
#     layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
# with tf.name_scope("layer4") as scope:
#     layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)
# with tf.name_scope("layer5") as scope:
#     layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)
# with tf.name_scope("layer6") as scope:
#     layer6 = tf.sigmoid(tf.matmul(layer5, W6) + b6)
# with tf.name_scope("layer7") as scope:
#     layer7 = tf.sigmoid(tf.matmul(layer6, W7) + b7)
# with tf.name_scope("layer8") as scope:
#     layer8 = tf.sigmoid(tf.matmul(layer7, W8) + b8)
# with tf.name_scope("layer9") as scope:
#     hypothesis = tf.sigmoid(tf.matmul(layer8, W9)+b9)
#
# with tf.name_scope("cost"):
#     cost = tf.reduce_mean(-Y*tf.log(hypothesis)-(1-Y)*tf.log(1-hypothesis))
#     tf.summary.scalar("cost", cost)
#
# with tf.name_scope("optimizer"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
#     train = optimizer.minimize(cost)
#
# with tf.name_scope("accuracy"):
#     predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#     tf.summary.scalar("accuracy", accuracy)
#
# with tf.Session() as sess:
#     merge_summary = tf.summary.merge_all()
#     write = tf.summary.FileWriter("./logs/aa")
#     write.add_graph(sess.graph)
#     sess.run(tf.global_variables_initializer())
#     for step in range(10001):
#         cost_val, train_val, summary_val = sess.run([cost, train, merge_summary], feed_dict={X: X_data, Y:Y_data})
#         write.add_summary(summary_val, global_step=step)
#         if step % 1000 == 0:
#             print(f"step = {step}, cost = {cost_val}")
#     a, p = sess.run([accuracy, predicted], feed_dict={X:X_data, Y:Y_data})
#     print(f"accuracy = {a}, predicted = {p}")
