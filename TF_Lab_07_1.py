# training / test, dataset, learning_rate, normalization

import tensorflow as tf
import numpy as np

x_data = [[1,2,1], [1,3,2], [1,3,4], [1,5,5], [1,7,5], [1,2,5], [1,6,6], [1,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

x_test = [[2,1,1], [3,1,2], [3,3,4]]
y_test = [[0,0,1], [0,0,1], [0,0,1]]
# 나중에 test할 때 사용하는 data

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 3])

W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

H = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H), axis=1))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(H, 1)
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, _ = sess.run([cost, W, train], feed_dict={X : x_data, Y : y_data})
        print(step, cost_val, W_val)

    print("Prediction :", sess.run(prediction, feed_dict={X : x_test}))
    print("Accuracy :", sess.run(accuracy, feed_dict={X : x_test, Y : y_test}))
    # test 확인!

#################################################################################

tf.set_random_seed(777)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)
# 자세히는 모르겠지만, data들을 자동적으로 0~1사이로 바꿔준다.
# 마치 softmax함수의 normalize 파라미터와 같다.

xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])
xy = MinMaxScaler(xy)
# 0과 1사이로 바꿔주는 애
print(xy)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(101):
        cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_data, Y : y_data})
        print("Step :", step, "Cost : ", cost_val, "\nPrediction\n", H_val)
