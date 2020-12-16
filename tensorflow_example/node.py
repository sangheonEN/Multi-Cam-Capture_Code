import tensorflow as tf

# Graph Build (node와 function 정의)
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
# print(f"node1 = {node1}, node2 = {node2}, node3 = {node3}")

# 뒤쪽 노드 출력
sess = tf.Session()
print("sess.run(node1, node2): {}".format(sess.run([node1, node2])))
print("sess.run(node3): {}".format(sess.run([node3])))

# Placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b : 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b : [3.5, 4.5]}))
