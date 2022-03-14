# 1. 일반적인 Linear Regression
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

# 2. placeholder를 이용한 Linear Regression placeholder는 sess.run()할때 변수에 값을 feed_dict = 변수명 : 넣고 싶은 데이터(리스트형식) 형식으로 넣을 수 있도록 변수를 초기화 하는 함수다.
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

# 3. Cost(loss) Function의 최소화(최적화) W 변화에 따른 cost(W) 변화 값
# import tensorflow as tf
# import matplotlib.pyplot as plt
#
# X = [1, 2, 3]
# Y = [1, 2, 3]
#
# # 가설의 모델링 Graph 그리기
# # W 변수 정의
# W = tf.placeholder(tf.float32)
#
# # 가설 정의 ()
# hypothesis = W*X
#
# # cost function 정의
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# # session run()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # W와 cost에 대한 시각화
# W_val = []
# cost_val = []
# for i in range(-30, 51):
#     feed_W = i * 0.1
#     curr_Cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
#     W_val.append(curr_W)
#     cost_val.append(curr_Cost)
#     print(f"step = {i}, W = {curr_W}, cost = {curr_Cost}")
#
# # 시각화 cost와 W
# plt.plot(W_val, cost_val)
# plt.show()

# 4. Gradient Decent algorithm 적용.     tf.train.GradientDescentOptimizer(learning_rate = 0.1) 함수 안쓰고 구현 정확히 무엇을 최소화 시키는지 확인해보자.
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# x_data = [1, 2, 3]
# y_data = [1, 2, 3]
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# W = tf.Variable(tf.random_normal([1]), name="weight")
#
# hypothesis = W * X
#
# cost = tf.reduce_mean(tf.square(W*X - Y))
#
# # optimizer --> 기울기 편차를 줄이면서 업데이트하는 것.
# learning_rate = 0.1
# gradient = tf.reduce_mean((W * X - Y) * X)
# decent = W - learning_rate * gradient
# update = W.assign(decent)
# # tf.train.GradientDescentOptimizer(learning_rate = 0.1) 이 함수와 역할이 동일하다. 함수에 내포된 기능으로 미분을 해주어서 계산된다.
#
# # session()
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for step in range(21):
#     sess.run(update, feed_dict={X : x_data, Y : y_data})
#     print(f"step : {step}, cost : {sess.run(cost, feed_dict={X : x_data, Y: y_data})}, W : {sess.run(W)}")

# 5. W = -3으로 초기 값을 주었을 때 tf.train.GradientDescentOptimizer변화에 대한 cost() 변화 값
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# X = [1, 2, 3]
# Y = [1, 2, 3]
#
# W = tf.Variable(-3.0)
#
# hypothesis = W * X
#
# cost = tf.reduce_mean(tf.square((W*X)-Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1) # Gradient Descent Optimizer 적용. W를 계속 업데이트하여서 줄여줌 기존 W - W 편미분한 값 반복
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(101):
#     print(f"step = {step}, W = {sess.run(W)}")
#     sess.run(train)

# 6. W = 5으로 초기 값을 주었을 때 tf.train.GradientDescentOptimizer변화에 대한 cost() 변화 값
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# X = [1, 2, 3]
# Y = [1, 2, 3]
#
# W = tf.Variable(5.0)
#
# hypothesis = W * X
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(101):
#     print(f"step = {step}, W = {sess.run(W)}")
#     sess.run(train)


# 7. Optional : compute_gradient : optimizer 에 Gradient함수를 적용한 후 곧 바로 minimize하는 것이 아니라 그 시점의 cost에 맞는 Gradient를 계산한 값을 가지고 변경할 수 있다.
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# X = [1, 2, 3]
# Y = [1, 2, 3]
#
# W = tf.Variable(5.0, tf.float32)
#
# hypothesis = W * X
#
# gradient = tf.reduce_mean((W * X - Y) * X) * 2       # 수식으로 직접 계산한 Gradient
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1)   # 함수에서 자동으로 계산된 Gradient
#
# # cost 값 대비 해서 Gradient를 gvs에 할당해라.
# # compute_gradient를 사용하면 자동으로 minimize되는 것 외로 사용자가 원하는 값으로 gradient를 변경하여 값을 줄 수 있다는 특징이 있다.
# gvs = optimizer.compute_gradients(cost)
# apply_gradient = optimizer.apply_gradients(gvs)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(101):
#     print(f"step = {step}, Gradient = {sess.run(gradient)}, W = {sess.run(W)}, gvs = {sess.run(gvs)}, cost = {sess.run(cost)}")
#     sess.run(apply_gradient)


# 8. 내가 원하는 데이터를 가지고 실습해보자.
# 야구공을 10번 던져 구속에 따라 점수가 주어지고 그 결과는 총합으로 측정하는 게임이 있다. 야구 선수가 얼마나 게임에 적응해 가는지 생각해보고 만약 1 ~ 10번 게임을 참가했을 때 총합 점수가 있다면, 13번째에는 얼마의 총합 점수가 될 지 예측해보자.
# X = 게임 경험 횟수 Y = 게임 회 당 총합 점수
# import tensorflow as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# X_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
# Y_data = [120.0, 130.0, 135.0, 140.0, 150.0, 160.0, 163.0, 170.0, 185.0, 200.0]
#
# W = tf.Variable(tf.random_normal([1]), name="Weight")
# b = tf.Variable(tf.random_normal([1]), name="bias")
# X = tf.placeholder(tf.float32, shape=[None])
# Y = tf.placeholder(tf.float32, shape=[None])
#
# hypothesis = W * X + b
#
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(cost)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for step in range(5001):
#     sess.run(train, feed_dict={X : X_data, Y : Y_data})
#     if step % 500 == 0:
#         print(f"step = {step}, W = {sess.run(W)}, b = {sess.run(b)}, cost = {sess.run(cost, feed_dict={X: X_data, Y : Y_data})}")
# print(sess.run(hypothesis, feed_dict={X : [13]}))
# 습득교훈 : learning_rate = 0.1 설정하니 학습이 되지 않았다. 하여, learning_rate에 대한 것도 공부해보자.














