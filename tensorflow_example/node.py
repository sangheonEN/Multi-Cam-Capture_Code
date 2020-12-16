import tensorflow as tf

# 앞쪽 노드 빌드
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
print(f"node1 = {node1}, node2 = {node2}, node3 = {node3}")

# 뒤쪽 노드 함수로 정의



# 뒤쪽 노드 출력