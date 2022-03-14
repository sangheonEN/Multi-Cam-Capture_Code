import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777) # for reproducibility

# Mnist_Data load
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Layer & Variable Definition

with tf.name_scope("Layer1"):
    X_DATA = tf.placeholder(tf.float32, shape=[None, 784])
    Y_DATA = tf.placeholder(tf.float32, shape=[None, 10])
    W1 = tf.Variable(tf.random_normal([784, 256]), name="weight1")
    b1 = tf.Variable(tf.random_normal([256]), name="bias1")
    logits1 = tf.matmul(X_DATA, W1)+b1
    layer1 = tf.nn.softmax(logits1)
    # tensorboard summary
    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([256, 50]), name="weight2")
    b2 = tf.Variable(tf.random_normal([50]), name="bias2")
    logits2 = tf.matmul(layer1, W2)+b2
    layer2 = tf.nn.softmax(logits2)
    # tensorboard summary
    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Layer2", layer2)

with tf.name_scope("Layer3"):
    num_calss = 10
    W3 = tf.Variable(tf.random_normal([50, num_calss]), name="weight3")
    b3 = tf.Variable(tf.random_normal([num_calss]), name="bias3")
    logits3 = tf.matmul(layer2, W3)+b3
    # hypothesis definition
    hypothesis = tf.nn.softmax(logits3)
    # tensorboard summary
    tf.summary.histogram("W3", W3)
    tf.summary.histogram("b3", b3)
    tf.summary.histogram("final layer, hypothesis", hypothesis)

# cost function definition
with tf.name_scope("Cost"):
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y_DATA)
    cost = tf.reduce_mean(cost_i)
    tf.summary.scalar("accuracy", cost)

# train method
with tf.name_scope("Train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)

# predicted & accuracy definition if) H(x) argmax and Y argmax result compare True False
predicted = tf.argmax(hypothesis, 1)
correct_train = tf.equal(predicted, tf.argmax(Y_DATA, 1))
accuracy = tf.reduce_mean(tf.cast(correct_train, tf.float32))
tf.summary.scalar("accuracy", accuracy)

# epoch, batch definition
epoch_num = 20
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# session
with tf.Session() as sess:
    # 터미널 명령어! : tensorboard --logdir=./경로
    # summary --> merge --> save file --> show the graph
    merge_summary = tf.summary.merge_all()
    write = tf.summary.FileWriter("./logs/mnist_data_logs_r0_01")
    write.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_num):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            cost_val, train_val, summary_val = sess.run([cost, train, merge_summary], feed_dict={X_DATA : batch_x, Y_DATA : batch_y})
            write.add_summary(summary_val, global_step=i)
        print(f"epoch = {epoch}, cost = {cost_val}")
    print("Learning Finish")

    # Accuracy
    Accuracy_val = sess.run(accuracy, feed_dict={X_DATA:mnist.test.images, Y_DATA:mnist.test.labels})
    print(f"Accuracy : {Accuracy_val}")

    # Predicted
    r = random.randint(0, mnist.test.num_examples -1)
    print(f"label = {sess.run(tf.argmax(mnist.test.labels[r : r+1], 1))}")
    print(f"prediction = {sess.run(tf.argmax(hypothesis, 1), feed_dict={X_DATA:mnist.test.images[r: r+1]})}")

    plt.imshow(
        mnist.test.images[r:r+1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest'
    )
    plt.show()









