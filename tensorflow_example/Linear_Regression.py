# 일반적인 Linear Regression
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]
#
# W = tf.Variable(tf.random_normal([1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')
#
# hypothesis = x_train * W + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))     # tf.reduce_mean() 평균 함수
#
# # optimize를 정의하고 최소화하기 위해 사용하는 GradientDescent
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
#
# # 실행 Session()
# sess = tf.Session()
# # 정의한 Variable 실행하기 위해 global_variables_initializer()
# sess.run(tf.global_variables_initializer())
#
# # 2000번 학습시키고 20번마다 결과 값을 출력해라. 결과값 : step, cost, W, b
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

# placeholder를 이용한 Linear Regression
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# # 변수 정의 W, b, x, y
# W = tf.Variable(tf.random_normal([1]), name="weight") # random_normal([1]) 1차원 배열 형태
# b = tf.Variable(tf.random_normal([1]), name="bias")
# X = tf.placeholder(tf.float32, shape=[None])          # shape=[] []에 따라서 차원이 설정되고 None은 개수 제한이 없다는 말임.
# Y = tf.placeholder(tf.float32, shape=[None])
#
# # 가설 설정
# hypothesis = X*W + b
#
# # cost/loss function 설정
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# # 최적화 Optimizer 설정, Train Data 설정
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.01)
# train = optimizer.minimize(cost)
#
# # 출력 전 sessison 설정, 변수 사용하기 위해 전역 선언
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # Fit the Line with new training data 학습 데이터(W,b,cost,train) 넣고 몇 번 학습할지? 결과를 어떤 변수를 출력할지?
# for step in range(2001):
#     cost_val, W_val, b_val, train_val = \
#         sess.run([cost, W, b, train],
#                  feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
#     if step % 20 == 0:
#         print(f"step : {step}, cost : {cost_val}, W : {W_val}, b : {b_val}, train : {train_val}")
#
# # Testing Model
# print(sess.run(hypothesis, feed_dict={X: [5]}))
# print(sess.run(hypothesis, feed_dict={X: [2.5, 10.1]}))
