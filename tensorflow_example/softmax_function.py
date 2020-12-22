# Multi_Classification with Softmax Function
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# numpy 사용 csv 파일 load
# tf.set_random_seed(777)
#
# xy = np.loadtxt('data-04-softmax.csv', delimiter=',', dtype= np.float32)
# x_data = xy[:, 0:4]
# y_data = xy[:, 4:7]
#
# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data, len(y_data))

# queue 사용 csv 파일 load
filename_queue = tf.train.string_input_producer(
    ['data-04-softmax.csv'], shuffle=False, name='filename_queue'
)

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)


record_default = [[0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_default)

# tf.io.decode_csv(
#     records, record_defaults, field_delim=',', use_quote_delim=True,
#     na_value='', select_cols=None, name=None
# )

# 옵션 값임으로 다 빼버리고, field_delim=',', use_quote_delim=True, na_value='', select_cols=None, name=None

train_x_batch, train_y_batch = tf.train.batch([xy[0:4], xy[4:7]], batch_size=8)

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

# shape 주의하자!! y = [1, 0, 0] 3개의 클래스
nb_class = 3
W = tf.Variable(tf.random_normal([4, nb_class]), name="Weight")
b = tf.Variable(tf.random_normal([nb_class]), name="bias")

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        x_data, y_data = sess.run([train_x_batch, train_y_batch])
        cost_val, train_val, hypothesis_val = sess.run([cost, train, hypothesis], feed_dict={X: x_data, Y: y_data})
        if step % 500 == 0:
            print(f"step = {step}, cost = {cost_val}, H(x) = {hypothesis_val}")
    coord.request_stop()
    coord.join(threads)
# csv 파일 불러오기 할려 했는데 안됨 오류 발생 tensorflow.python.framework.errors_impl.OutOfRangeError: FIFOQueue '_2_batch/fifo_queue' is closed and has insufficient elements (requested 8, current size 0)
# 	 [[node batch (defined at \ProgramData\Anaconda3\envs\imagedataprocessing\lib\site-packages\tensorflow_core\python\framework\ops.py:1748) ]]

# Multi_Classification with Softmax Function. binary 0 or 1 2가지의 예측 값을 얻는게 아닌 N개의 예측 값을 얻기 위해 사용. softmax 함수를 활용해서 Y^ 예측치를 확률로 표현할 수 있다. 따라서, COST함수를 만들 수 있음.
# import tensorflow as tf
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# x_data = [[1,2,1,15], [20, 1, 3, 2], [30, 1, 3, 4], [4, 20, 5, 5], [1, 30, 5, 5], [1, 2, 50, 6], [1, 6, 60, 6], [1, 7, 7, 70]]
# y_data = [[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
#
# X = tf.placeholder(tf.float32, shape=[None, 4])
# Y = tf.placeholder(tf.float32, shape=[None, 4])
#
# num_class = 4
# W = tf.Variable(tf.random_normal([4, num_class]), name="weight")
# b = tf.Variable(tf.random_normal([num_class]), name="bias")
#
# hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)
#
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis)))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(10001):
#         cost_val, train_val, hypothesis_val = sess.run([cost, train, hypothesis], feed_dict={X: x_data, Y: y_data})
#         if step % 500 == 0:
#             print(f"step : {step}, cost : {cost_val}, H(x) : \n{hypothesis_val}")
#     a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
#     print(f"X = [1, 11, 7, 9]\nY = {a}, \narg_max(Y = 1인 index) = {sess.run(tf.arg_max(a, 1))}")  # arg_max(가설 예측 치, 축) 축 = 1 이라는 것은 Y^ 예측치 (Y^1, Y^2, Y^3) 증 가장 1에 가까운 INDEX를 출력해라는 것!
#     all = sess.run(hypothesis, feed_dict={X : [[111, 2, 3, 1],
#                                                [2, 111, 3, 5],
#                                                [14, 1, 120, 2],
#                                                [1, 2, 3, 133]]})
#     print(f"all Y^ = {all}, \nall arg_max = {sess.run(tf.arg_max(all, 1))} ")

# Fancy Softmax