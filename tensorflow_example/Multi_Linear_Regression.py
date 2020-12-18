# Multi Linear Regression Matrix를 활용하지 않은 선형 회귀 학습 예측 모형 X 데이터가 무수히 많아지면 소스코드가 그만큼 줄이 늘어남..
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# X1_data = [73., 93., 89., 96., 73.]
# X2_data = [80., 88., 91., 98., 66.]
# X3_data = [75., 93., 90., 100., 70.]
# Y_data = [152., 185., 180., 196., 142.]
#
# X1 = tf.placeholder(tf.float32)
# X2 = tf.placeholder(tf.float32)
# X3 = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# W1 = tf.Variable(tf.random_normal([1]), name="Weight")
# W2 = tf.Variable(tf.random_normal([1]), name="Weight")
# W3 = tf.Variable(tf.random_normal([1]), name="Weight")
# b = tf.Variable(tf.random_normal([1]), name="bias")
#
# hypothesis = X1*W1 + X2*W2 + X3*W3 + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(15001):
#     cost_val, hy_val, train_val = sess.run([cost, hypothesis, train], feed_dict={X1 : X1_data, X2 : X2_data, X3 : X3_data, Y : Y_data})
#     if step % 100 == 0:
#         print(f"STEP : {step}, cost : {cost_val}, hypothesis 예측값 : {hy_val}")
# print(f"X1 = 50, X2 = 80, X3 = 100일때, 최종 점수 Y = {sess.run(hypothesis, feed_dict={X1 : 50, X2 : 80, X3 : 100})}")

# Multi Linear Regression Matrix를 활용한 다중 선형 회귀 학습 예측 모형
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# X_data = [[73., 80., 75.],
#           [93., 88., 93.],
#           [89., 91., 90.],
#           [96., 98., 100.],
#           [73., 66., 70.]]
#
# Y_data = [[152.],
#           [185.],
#           [180.],
#           [196.],
#           [142.]]
#
# X = tf.placeholder(tf.float32, shape=[None, 3])               # 5개를 가지고 있지만 필요에 따라 늘릴 수 있으니 None(N개)
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([3, 1]), name="Weight")      # X * W = Y  [5,3] * [] = [5,1]       W = [3,1]
# b = tf.Variable(tf.random_normal([1]), name="bias")
#
# hypothesis = tf.matmul(X, W) + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(5001):
#     cost_val, hypothesis_val, train_val = sess.run([cost, hypothesis, train], feed_dict={X : X_data, Y : Y_data})
#     if step % 200 == 0:
#         print(f"step = {step}, cost = {cost_val}, 예측 값 = {hypothesis_val}")

# Numpy Array[] Slicing 적용
# import tensorflow as tf
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# Raw_data = [[73., 80., 75., 152.],
#           [93., 88., 93., 185.],
#           [89., 91., 90., 180.],
#           [96., 98., 100., 196.],
#           [73., 66., 70., 142.]]
#
# array = np.array(Raw_data)
#
# X_data = array[:, 0:3]
# Y_data = array[:, 3:4]
# # X_data = array[:, 0:-1]
# # Y_data = array[:, [-1]]
# print("x = ", X_data.shape,"\n",X_data,"\n",len(X_data))
# print("y = ", Y_data.shape,"\n", Y_data,"\n",len(Y_data))
#
# X = tf.placeholder(tf.float32, shape=[None, 3])       # shape[행의 수, 열의 수]
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([3, 1]))             # X * W = Y     ->     5,3 * 3,1 = 5,1
# b = tf.Variable(tf.random_normal([1]))
#
#
# hypothesis = tf.matmul(X, W) + b                                # Multi Linear Regression 함수 = tf.matmul(X, W)
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(3001):
#     cost_val, hypothesis_val, train_val = sess.run([cost, hypothesis, train], feed_dict={X : X_data, Y : Y_data})
#     if step % 200 == 0:
#         print(f"step: {step}, cost = {cost_val}, \n 예측 값 = {hypothesis_val}")
# print(f"My Score x1 : 100, x2 :70, x3 : 80 \nFinal Score : {sess.run(hypothesis, feed_dict={X : [[100, 70, 80]]})}")

# csv File 불러오기
# import tensorflow as tf
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# filename_queue = tf.train.string_input_producer(
#     ['data-01-test-score.csv'], shuffle=False, name='filename_queue'  # queue에 csv파일을 저장
# )
#
# reader = tf.TextLineReader()
# key, value = reader.read(filename_queue)
#
# record_default = [[0.], [0.], [0.], [0.]]                     # csv의 각각의 데이터 type 설정 float
# xy = tf.decode_csv(value, record_defaults=record_default)     # csv로 디코더
#
# train_x_batch, train_y_batch = tf.train.batch([xy[0:3], xy[3:4]], batch_size=10)
#
# # X_data = array[:, 0:-1]
# # Y_data = array[:, [-1]]
# # print("x = ", train_x_batch.shape,"\n",train_x_batch,"\n",len(train_x_batch))
# # print("y = ", train_y_batch.shape,"\n", train_y_batch,"\n",len(train_y_batch))
#
# X = tf.placeholder(tf.float32, shape=[None, 3])       # shape[행의 수, 열의 수]
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([3, 1]), name="Weight")             # X * W = Y     ->     5,3 * 3,1 = 5,1
# b = tf.Variable(tf.random_normal([1]), name="bias")
#
#
# hypothesis = tf.matmul(X, W) + b                                # Multi Linear Regression 함수 = tf.matmul(X, W)
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
# for step in range(3001):
#     x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
#     cost_val, hypothesis_val, train_val = sess.run([cost, hypothesis, train], feed_dict={X : x_batch, Y : y_batch})
#     if step % 200 == 0:
#         print(f"step: {step}, cost = {cost_val}, \n 예측 값 = {hypothesis_val}")
# print(f"My Score x1 : 100, x2 :70, x3 : 80 \nFinal Score : {sess.run(hypothesis, feed_dict={X : [[100, 70, 80]]})}")
#
# coord.request_stop()
# coord.join(threads)


