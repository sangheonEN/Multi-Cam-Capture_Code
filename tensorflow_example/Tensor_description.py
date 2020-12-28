import tensorflow as tf
import numpy as np

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
#
# # 1차원 줄이고 1행열로 나열
# result = tf.reshape(t, shape=[-1, 3])
# # 차원 줄이지 말고 1행열로 나열
# result2 = tf.reshape(t, shape=[-1, 1, 3])
# # 스퀴즈 한 차원 줄여주고 나열
# squeeze1 = tf.squeeze(result2)
# # expand_dims 차원 늘리기
# expand_dims1 = tf.expand_dims(squeeze1, 1)
# # one_hot 배열 중 가장 큰 수 1로 만들기. 나머지는 0
# Hot = tf.one_hot([[0], [1], [2], [3], [4]], depth=5)  # depth : 가장 깊숙한 차원의 수 (1, 2, 3, 4, 5) 콤마의 개수! one_hot 함수를 사용하면 1차원이 더 생겨 출력됨.
# # one_hot reshape 차원 줄이기!
# tf.reshape(Hot, shape=[-1, 1])
#
# with tf.Session() as sess:
#     # print("기존의 행렬 : ", t)
#     # print("1차원 줄이고 1행열로 나열", result.eval(session=sess))
#     # print("차원 줄이지말고 1행열로 나열", result2.eval(session=sess))
#     # print("squeeze : 차원을 1차원으로 쭉 펴준다.", squeeze1.eval(session=sess))
#     # print("expand_dims : 차원을 늘림.", expand_dims1.eval(session=sess))
#     print("[[0], [1], [2], [3], [4]]의 rank가 2이지만 one_hot하면 rank가 3이됨.", Hot.eval(session=sess))
#     print("[[0], [1], [2], [3], [4]]의 one_hot reshape rank가 3에서 2로 줄이기.", tf.reshape(Hot, shape=[-1, 5]).eval(session=sess)) # shape=[-1, 5] -1은 행을 어떻게 나타내는지에 대한 명령어 인데 원하는 개수로 늘려라는 뜻임.
#     print(f"float -> cast tf.int32 {tf.cast([1.3, 23.4, 12.3], tf.int32).eval(session=sess)}")
#     print(f"boolian -> cast tf.int32 {tf.cast([True, False, 1==1, 1==0], tf.int32).eval(session=sess)}")
# # axis와 stack과의 관계
# x = [1, 4]
# y = [2, 3]
# z = [10, 2]
# with tf.Session() as sess:
#     print(f"{tf.stack([x, y, z], axis=1).eval(session=sess)}")
#     # print(f"{tf.stack([x, y, z], axis=1).eval(session=sess)}")
# # ones_like, zeros_like
# x1 = [[0, 1, 2],
#       [2, 1, 0]]
# with tf.Session() as sess:
#     print(f"{tf.ones_like(x1).eval(session=sess)}")
#     print(f"{tf.zeros_like(x1).eval(session=sess)}")
#
# # zip 여러개(복수개)의 tensor를 가지고 있을 때 한번에 추출하고 싶다.
# for x, y in zip([1, 2, 3], [4, 5, 6]):
#     print(f"x : {x}, y : {y}")




# 함수와 axis와의 관계
# argmax axis = 0 열 단위로 계산 후 큰 인덱스의 값을 출력 axis = 1 행 단위로 계산 후 큰 인덱스의 값을 출력
x1 = [[0, 1, 2],
      [2, 1, 0],
      [3, 4, 10]]
y1 = [[2, 3, 2],[4, 5, 3], [7, 8, 4]]
# with tf.Session() as sess:
#     print(f"{tf.argmax(x1, axis=0).eval(session=sess)}")
#     print(f"{tf.argmax(x1, axis=1).eval(session=sess)}")

# reduce_mean axis = 0 열단위로 평균 계산 axis = 1 행단위로 평균 계산
# with tf.Session() as sess:
#     print(f"{tf.reduce_mean(x1, axis=0).eval(session=sess)}")
#     print(f"{tf.reduce_mean(x1, axis=1).eval(session=sess)}")

# reduce_sum axis = 0 열단위로 총합 계산 axis = 1 행단위로 총합 계산
# with tf.Session() as sess:
#     print(f"{tf.reduce_sum(x1, axis=0).eval(session=sess)}")
#     print(f"{tf.reduce_sum(x1, axis=1).eval(session=sess)}")

# stack
with tf.Session() as sess:
    print(f"axis = 0 {tf.stack([x1, y1], axis=0).eval(session=sess)}")
    print(f"axis = 1 {tf.stack([x1, y1], axis=1).eval(session=sess)}")
