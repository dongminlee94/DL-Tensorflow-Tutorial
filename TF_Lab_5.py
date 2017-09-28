import tensorflow as tf
import numpy as np

# x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
# # 첫번째열은 x1, 두번째열은 x2
# y_data = [[0], [0], [0], [1], [1], [1]]
# # binary Classification 이므로 0 or 1로만 주어진다.
#
# # ex) 만약에 첫번째 열과 두번째 열이 공부한 시간
# # y데이터는 0은 Fail, False, 1은 Pass, True라고 하자
#
# X = tf.placeholder(tf.float32, shape=[None, 2])
# Y = tf.placeholder(tf.float32, shape=[None, 1])
# # 앞으로 placeholder만들 때 shape도 같이 만들기
#
# W = tf.Variable(tf.random_normal([2, 1]))
# b = tf.Variable(tf.random_normal([1]))
# # b의 shape은 항상 Y의 개수와 같다.
#
# H = tf.sigmoid(tf.matmul(X, W) + b)
# # sigmoid function 적용
# cost = -tf.reduce_mean(Y*tf.log(H) + (1 - Y)*tf.log(1 - H))
#
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#
# predicted = tf.cast(H > 0.5, dtype=tf.float32)
# # 기준점을 하나 만든다. 0.5가 그 기준점이 된다. 0.8은 pass 1, 0.1은 fail 0
# # float32로 0.5를 캐스팅하면 F는 0, T는 1이 된다.
# # cast라는 것은 텐서를 새로운 자료형으로 변환한다.
# # dtype을 적어주면 그 dtype으로 형변환을 한다.
#
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# # 우리가 예상한 predicted와 Y가 동일한지 캐스팅해서 정확한지 안한지 보는 것
# # predicted와 Y가 동일하면 1, 그렇지 않으면 0
#
#
# # with as를 쓰면 인터프리터가 자동으로 구문이 끝나면 마무리를 해준다.
# # 화장실을 갔다 오면 지퍼를 올리는 것 처럼
# # 파일을 처리하고자 할 경우에 with as가 자동으로 열고 닫으니까 주로 많이 쓰인다.
# # enter()와 exit()를 호출한다.
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         cost_val, _ = sess.run([cost, train], feed_dict={X : x_data, Y : y_data})
#         if step % 200 == 0:
#             print(step, cost_val)
#
#     h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
#     print(h, c, a)
#
# #####################################################################################
#
# # # 직접 실제 데이터를 읽어서 해보자!
# # # Classifying diabetes 당뇨병을 예측하는 데이터를 읽어서 학습해보자.
#
# # tf.set_random_seed(777)
# #
# # xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
# # x_data = xy[:, 0:-1]
# # y_data = xy[:, [-1]]
# #
# # print(x_data.shape, y_data.shape)
# # print(x_data, y_data)
# #
# # X = tf.placeholder(tf.float32, shape=[None, 8])
# # Y = tf.placeholder(tf.float32, shape=[None, 1])
# #
# # W = tf.Variable(tf.random_normal([8, 1]))
# # b = tf.Variable(tf.random_normal([1]))
# #
# # H = tf.sigmoid(tf.matmul(X, W) + b)
# # cost = tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))
# # train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# #
# # predicted = tf.cast(H > 0.5, dtype=tf.float32)
# # accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #
# #     feed = {X : x_data, Y : y_data}
# #     for step in range(10001):
# #         sess.run(train, feed_dict=feed)
# #         if step % 200 == 0:
# #             print(step, sess.run(cost, feed_dict=feed))
# #
# #     h, c, a = sess.run([H, predicted, accuracy], feed_dict=feed)
# #     print(h, c, a)

################################################################################################

# code practice 1

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 1])

W = tf.Variable(tf.random_normal([2, 1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.sigmoid(tf.matmul(X, W) + b)
cost = -tf.reduce_mean(Y * tf.log(H) + (1 - Y) * tf.log(1 - H))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(H > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X : x_data, Y : y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
    print(h, c, a)

tf.set_random_seed(777)
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data)
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, [None, 8])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([8, 1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.sigmoid(tf.matmul(X, W) + b)
cost = tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
predicted = tf.cast(H > 0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X : x_data, Y : y_data})

    h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
    print(h, c, a)








