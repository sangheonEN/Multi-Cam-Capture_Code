import tensorflow as tf

# t1 = tf.constant([1, 2, 3, 4])
#
# t2 = tf.constant([[1, 2],
#                  [3, 4]])

# matrix1 = tf.constant([[1., 2.], [3., 4.]])
# matrix2 = tf.constant([[1.], [2.]])
#
# with tf.Session() as sess:
#     print(matrix1.eval(session=sess))
#     print(matrix2.eval(session=sess))
#     print(matrix1.shape)
#     print(matrix2.shape)
#     print(tf.matmul(matrix1, matrix2).eval(session=sess))

# t = np.array([[[0., 1., 2.],
#                [3., 4., 5.]],
#               [[6., 7., 8.],
#                [9., 10., 11.]]])
# print(t.shape)

# 1차원 줄이고 1행열로 나열
result = tf.reshape(t, shape=[-1, 3])
# 차원 줄이지 말고 1행열로 나열
result2 = tf.reshape(t, shape=[-1, 1, 3])
# 스퀴즈 한 차원 줄여주고 나열
squeeze1 = tf.squeeze(result2)
# expand_dims 차원 늘리기
expand_dims1 = tf.expand_dims(squeeze1, 1)
# one_hot 배열 중 가장 큰 수 1로 만들기. 나머지는 0
Hot = tf.one_hot([1, 2, 3, 4], depth=3)

with tf.Session() as sess:
    # print("기존의 행렬 : ", t)
    # print("1차원 줄이고 1행열로 나열", result.eval(session=sess))
    # print("차원 줄이지말고 1행열로 나열", result2.eval(session=sess))
    # print("squeeze : 차원을 1차원으로 쭉 펴준다.", squeeze1.eval(session=sess))
    # print("expand_dims : 차원을 늘림.", expand_dims1.eval(session=sess))
    print("one_hot", Hot.eval(session=sess))

