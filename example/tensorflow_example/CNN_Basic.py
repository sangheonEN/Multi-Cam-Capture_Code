import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)

# filter의 갯수 늘리기. parameter를 추가하면 됨.
# weight = tf.constant([[[[1.]], [[1.]]],
#                        [[[1.]], [[1.]]]])
weight = tf.constant([[[[1., -1., 5.]], [[1., -1., 5.]]],
                       [[[1., -1., 5.]], [[1., -1., 5.]]]])

# image shape = (1, 3, 3, 1) --> 1개의 이미지를 가지고, 3 * 3 size의 이미지 데이터를 가지고, Gray 1채널 컬러로 표현하겠다.
# weight(filter) shape = (2, 2, 1, 1) --> 2*2 size의 filter를 만들고, Gray 1차원 컬러로 표현하고, 1개의 filter를 만들겠다.
# conv2d = tf.nn.conv2d(image,weight,strides=[1, 1, 1, 1], padding='VALID')
print(f"image shape : {image.shape} \nweight shape : {weight.shape}")

# Padding = "SAME" 입력데이터와 출력데이터 사이즈를 같게 함.
# ZERO Padding 되어서 사이즈를 같게 만듬.
conv2d = tf.nn.conv2d(image,weight,strides=[1, 1, 1, 1], padding='SAME')
conv2d_image = conv2d.eval()

# conv2d_image.shape = (1, 3, 3, 3) --> Gray 1채널 컬러로 표현하고, 3 * 3 size 이며, filter가 3개니 conv도 3개이다.
print(f"conv2d image shape = {conv2d_image.shape}")
conv2d_image = np.swapaxes(conv2d_image, 0, 3)

# max-Pooling ksize=[batchsize, x범위, y범위, channel], strides =[batchsize, x범위, y범위, channel]
pool = tf.nn.max_pool(conv2d_image, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
print(f"pool shape = {pool.shape}")
print("pool finish")
pool_image = np.swapaxes(pool.eval(), 0, 3)

for i, one_img in enumerate(conv2d_image):
    print("conv value")
    print(one_img.reshape(3, 3))
    plt.subplot(1, 3, i+1), plt.imshow(one_img.reshape(3, 3), cmap="gray")
plt.show()
for i, two_img in enumerate(pool_image):
    print("pool value")
    print(two_img.reshape(2, 2))
    plt.subplot(1, 3, i+1), plt.imshow(two_img.reshape(2, 2), cmap="gray")


plt.show()
# plt.imshow(image.reshape(3, 3), cmap="Greys")
# plt.show()

# print(f"pool value = {pool.eval()}")

