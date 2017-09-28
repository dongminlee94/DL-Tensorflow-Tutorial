import tensorflow as tf

hello = tf.constant('Hello, TensorFlow')
# 어떠한 그래프 안에 노드가 하나 있고 그 안에 있는 글자가 헬로우 텐서플로
# constant 텐서하나 만들기(리스트도 가능!)

# sess = tf.Session()
# 하나의 객체를 만들 때 사용한다. 실행할려면 항상 있어야한다.

# print(sess.run(hello))
# 헬로우라는 노드를 실행할 것이다.
# 'b' 라는것이 나오면 byte string이다. 신경안써도 된다.

#######################################################

# 1. Build graph(tensors)
node1 = tf.constant(3.0, tf.float32)
# 데이터 타입도 바로 줄 수 있다.
node2 = tf.constant(4.0, tf.float32) # also tf.float32 implicitly
node3 = tf.add(node1, node2)
# node3 = node1 + node2

# print("node1 : ", node1, "node 2 : ", node2)
# print("node3 : ", node3)
# 여기까지만 출력을 하게 되면 그저 그래프에 있는 하나의 노드야! 라고 보여주기만 한다. 따로 값이 나오지 않는다.


# 2. feed data and run graph sess.run(op)
sess = tf.Session()
# 실행하기 위해서는 반드시 있어야한다.
# print("sess.run(node1, node2) : ", sess.run([node1, node2]))
# print("sess.run(node3) : ", sess.run([node3]))

#####################################################################################

a = tf.placeholder(tf.float32)
#placeholder라고 입력변수를 하나 만든다.
b = tf.placeholder(tf.float32)
adder_node = a + b # +는 add

print(sess.run(adder_node, feed_dict={a : 3, b: 4.5}))
#feed_dict라는게 input data
print(sess.run(adder_node, feed_dict={a : [1,3], b: [2,4]}))
# sess = tf.Session()는 한번만 하면되는구만
# tensorflow는 똑똑해서 배열로 해도 a + b를 알아듣는다.
# 이와같이 우리는 a + b라는 그래프를 하나 만들어 두고 placeholder라는 것을 만들어서
# feed_dict를 통해 원하는 입력 값을 출력해낼 수 있다.

###############################################################################33

# code practice 1

# hello = tf.constant('Hello, Tensorflow')
# # print(sess.run(hello))
#
# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.5)
# node3 = node1 + node2
#
# sess = tf.Session()
# # print("sess.run(node3) : ", sess.run(node3))
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b
#
# print(sess.run(adder_node, feed_dict={a : 3, b : 4.5}))
# print(sess.run(adder_node, feed_dict={a : [3, 4, 5], b : [1, 5, 6]}))

#######################################################################################

# code practice 2

# Hello = tf.constant('Hello, Tensorflow')
#
# n1 = tf.constant(3.0, tf.float32)
# n2 = tf.constant(4.5, tf.float32)
# n3 = n1 + n2
#
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# c = a + b
#
# sess = tf.Session()
# print(sess.run(Hello))
# print(sess.run(n3))
# print(sess.run(c, feed_dict={a : 3.0, b : 4.5}))
# print(sess.run(c, feed_dict={a : [1, 2, 3], b : [4, 5, 6]}))






