# Logistic Regression 알고리즘 : Classification에서 가장 정확도가 높은 알고리즘
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # 변수 정의
# X_Data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# Y_Data = [[0], [0], [0], [1], [1], [1]]
#
# X = tf.placeholder(tf.float32, shape=[None, 2])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([2, 1]), name="Weight")   # W shape = [2, 1]      X * W = Y -> [2, 1] * [2, 1] = [1, 1]
# b = tf.Variable(tf.random_normal([1]), name="bias")
#
# # tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#
# cost = -tf.reduce_mean(Y * tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
#
# # 가설 Hypothesis 값 0.5 기준으로 예측 된 결과 값을 True or False 구하고 float으로 dtype 변환
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# # 실제 값 Y와 가설 Hypothesis로 예측 된 결과 값인 predicted 값이 같은 지 다른지 비교하여 최종 정확도를 구한다.
# # accuracy는 실제 값과 예측 값이 같은지 아닌지 n번 예측 했다고 가정하면, 그 빈도의 평균을 구함.
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# # session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         cost_val, train_val = sess.run([cost, train], feed_dict={X: X_Data, Y : Y_Data})
#         if step % 200 == 0:
#             print(f"step = {step}, cost = {cost_val}")
#     # 가설 값, 가설을 기반한 실제 값, 정확도 출력
#     h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:X_Data, Y:Y_Data})
#     print(f"hypothesis = {h}, predicted = {p}, accuracy = {a}")

# 실제 예제인 당뇨병을 앓고 있는 사람과 아닌 사람의 기초 의료 데이터이다. 제시된 parameter를 통해 내 환자가 당뇨병에 걸렸을 지 안 걸렸을 지 테스트를 해보는 실습을 해보자. file = data-03-diabetes.csv
# import tensorflow as tf
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# X_Data = xy[:, 0:-1]
# Y_Data = xy[:, [-1]]
#
# X = tf.placeholder(tf.float32, shape=[None, 8])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# W = tf.Variable(tf.random_normal([8, 1]), name="Weight")
# b = tf.Variable(tf.random_normal([1]), name="bias")
#
# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
#
# cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
# train = optimizer.minimize(cost)
#
# predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(Y, predicted), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for step in range(10001):
#         cost_val, train_val = sess.run([cost, train], feed_dict={X:X_Data, Y:Y_Data})
#         if step % 500 == 0:
#             print(f"step = {step}, cost = {cost_val}, train = {train_val}")
#     h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:X_Data, Y:Y_Data})
#     print(f"hypothesis = {h}\npredicted = {p}\nacurracy = {a}\n")
#
#     print(f"""if I spec is x1 = -0.129333, x2 = 0.577333, x3 = 0.172733, x4 = -0.99999, x5 = 0, x6 = 0.122, x7 = -0.033, x8 = -0.77
#                 \n hypothesis = {sess.run(hypothesis, feed_dict={X : [[-0.129333, 0.577333, 0.172733, -0.999999, 0., 0.1222, -0.033, -0.7777]]})}
#                 \n result = {sess.run(predicted, feed_dict={X : [[-0.129333, 0.577333, 0.172733, -0.999999, 0., 0.1222, -0.033, -0.7777]]})}""")
















