import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) # for reproducibility

# Mnist_Data load
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# CNN START
sess = tf.InteractiveSession()

# hyper parameters
learning_rate = 0.001
train_epoch = 20
batch_size = 100

# X_DATA, Y_DATA, image, weight, conv definition
# X = tf.placeholder(tf.float32, shape=[None, 784])
# Y = tf.placeholder(tf.float32, shape=[None, 10])
img = mnist.train.images[0].reshape(-1, 28,28, 1) # n개 만큼의 28*28 사이즈인 데이터를 1차원 gray image를 할당

# Weight definition
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))

# CONV strides=[1, 2, 2, 1], padding="SAME"
conv1 = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding="SAME")

# 변수 초기화
sess.run(tf.global_variables_initializer())

print(f"conv1 = {conv1}")
conv1_img = conv1.eval()
conv1_img = np.swapaxes(conv1_img, 0, 3)

for i, one_img in enumerate(conv1_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(14, 14), cmap="gray")
plt.show()

# Max Pooling
pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
print(f"pool {pool}, pool shape = {pool.shape}")

# 변수 초기화
sess.run(tf.global_variables_initializer())

pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)

for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(7,7), cmap="gray")
plt.show()

