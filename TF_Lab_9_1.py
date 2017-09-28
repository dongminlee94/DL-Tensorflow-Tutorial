import tensorflow as tf
import numpy as np

# x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
# y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# W = tf.Variable(tf.random_normal([2,1]))
# b = tf.Variable(tf.random_normal([1]))
#
# H = tf.sigmoid(tf.matmul(X, W) + b)
# cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# predicted = tf.cast(H > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         sess.run(train, feed_dict={X : x_data, Y : y_data})
#         if step % 100 == 0:
#             print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run(W))
#
#     h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
#     print(h, c, a)

# 이렇게 하면 학습을 정확히 하지 못한다.
####################################################################

# x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
# y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# W1 = tf.Variable(tf.random_normal([2,2]))
# # x의 input이 두개라서 앞에꺼는 2
# # x의 output(출력)을 몇개로 할껀가? 두개로 할것이다! 뒤에꺼 2
# b1 = tf.Variable(tf.random_normal([2]))
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
# # layer1이 하나의 H 역할을 한다.
#
# W2 = tf.Variable(tf.random_normal([2,1]))
# # 앞에 W1의 뒤에가 2였기 때문에 앞에를 2로 적는다.
# # Y의 output은 하나로 나와야하기 때문에 1
# b2 = tf.Variable(tf.random_normal([1]))
#
# H = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# predicted = tf.cast(H > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(5001):
#         sess.run(train, feed_dict={X : x_data, Y : y_data})
#         if step % 100 == 0:
#             print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run([W1, W2]))
#
#     h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
#     print(h, c, a)

###############################################################333

# # 더 wide하고 deep하고 해보겠다.
#
# x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
# y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#
# X = tf.placeholder(tf.float32, [None, 2])
# Y = tf.placeholder(tf.float32, [None, 1])
#
# W1 = tf.Variable(tf.random_normal([2, 10]))
# b1 = tf.Variable(tf.random_normal([10]))
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random_normal([10, 10]))
# b2 = tf.Variable(tf.random_normal([10]))
# layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
#
# W3 = tf.Variable(tf.random_normal([10, 10]))
# b3 = tf.Variable(tf.random_normal([10]))
# layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
#
# W4 = tf.Variable(tf.random_normal([10, 1]))
# b4 = tf.Variable(tf.random_normal([1]))
# H = tf.sigmoid(tf.matmul(layer3, W4) + b4)
#
# cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# predicted = tf.cast(H > 0.5, tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(10001):
#         sess.run(train, feed_dict={X : x_data, Y : y_data})
#         if step % 100 == 0:
#             print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run([W1, W2]))
#
#     h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
#     print(h, c, a)

################################################################################333

# code practice 1

# x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
# y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
#
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
#
# W1 = tf.Variable(tf.random_normal([2,2]))
# b1 = tf.Variable(tf.random_normal([2]))
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable(tf.random_normal([2,1]))
# b2 = tf.Variable(tf.random_normal([1]))
# H = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# predicted = tf.cast(H > 0.5, dtype=tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for step in range(5001):
#         sess.run(train, feed_dict={X : x_data, Y : y_data})
#         if step % 100 == 0:
#             print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run([W1, W2]))
#
#     h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
#     print(h, c, a)

##############################################################################33

# Exercise
# Wide and Deep NN for MNIST

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.Variable(tf.random_normal([784, 50]))
b1 = tf.Variable(tf.random_normal([50]))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([50, 10]))
b2 = tf.Variable(tf.random_normal([10]))
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 10]))
b3 = tf.Variable(tf.random_normal([10]))
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 10]))
b4 = tf.Variable(tf.random_normal([10]))
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([10, 10]))
b5 = tf.Variable(tf.random_normal([10]))
H = tf.sigmoid(tf.matmul(layer4, W5) + b5)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(H, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X : batch_xs, Y : batch_ys})
            avg_cost += c / total_batch
        print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X : mnist.test.images, Y : mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    # 랜덤한 숫자 하나를 읽어온다.
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    # mnist.test.labels[r:r+1] 이게 one_hot에서 끌고온 숫자가 된다.
    print("Prediction : ", sess.run(tf.argmax(H, 1), feed_dict={X : mnist.test.images[r: r+1]}))
    plt.imshow(mnist.test.images[r : r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

