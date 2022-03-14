# Numpy Array[] Slicing 적용
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

Raw_data = [[73., 80., 75., 152.],
          [93., 88., 93., 185.],
          [89., 91., 90., 180.],
          [96., 98., 100., 196.],
          [73., 66., 70., 142.]]

array = np.array(Raw_data)

X_data = array[:, 0:3]
Y_data = array[:, 3:4]
# X_data = array[:, 0:-1]
# Y_data = array[:, [-1]]
print("x = ", X_data.shape,"\n",X_data,"\n",len(X_data))
print("y = ", Y_data.shape,"\n", Y_data,"\n",len(Y_data))

X = tf.placeholder(tf.float32, shape=[None, 3])       # shape[행의 수, 열의 수]
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([3, 1]))             # X * W = Y     ->     5,3 * 3,1 = 5,1
b = tf.Variable(tf.random_normal([1]))


hypothesis = tf.matmul(X, W) + b                                # Multi Linear Regression 함수 = tf.matmul(X, W)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate= 1e-5)
train = optimizer.minimize(cost)

# saver
SAVE_DIR = "logs/model"
saver = tf.train.Saver()
checkpoint_path = os.path.join(SAVE_DIR, "model")
ckpt = tf.train.get_checkpoint_state(SAVE_DIR)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # checkpoint file과 가장 최근 checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(f"test Cost : {cost.eval(feed_dict={X : X_data, Y : Y_data})}")
        print(f"pred value : {hypothesis.eval(feed_dict={X : X_data, Y : Y_data})[0]}")
        print(f"real value : {Y_data[0]}")
        sess.close()
        exit()

    for step in range(3001):
        cost_val, hypothesis_val, train_val = sess.run([cost, hypothesis, train], feed_dict={X : X_data, Y : Y_data})
        if step % 200 == 0:
            saver.save(sess, checkpoint_path, global_step = step)
            print(f"step: {step}, cost = {cost_val}, \n 예측 값 = {hypothesis_val}")
    print(f"My Score x1 : 100, x2 :70, x3 : 80 \nFinal Score : {sess.run(hypothesis, feed_dict={X : [[100, 70, 80]]})}")


