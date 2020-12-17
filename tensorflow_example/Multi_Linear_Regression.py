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



