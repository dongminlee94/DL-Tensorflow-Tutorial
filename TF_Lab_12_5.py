# RNN with time series data(stock data)

'''
This script shows how to predict stock prices using a basic RNN
'''

import tensorflow as tf
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 5
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
# 시간 순으로 만들기위해서 한번 뒤집는다.
xy = MinMaxScaler(xy)
# 값이 들쑥날쑥하니까 normalization을 해준다.
x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

'''
Example)
[[ 0.11324289  0.1142411   0.12394871  0.12772408  0.12007584]
 [ 0.10721065  0.11567471  0.12236416  0.10521648  0.1129937 ]
 [ 0.10639469  0.11955947  0.12039798  0.11555169  0.13045145]
 [ 0.10592847  0.10669713  0.10775088  0.22175463  0.09533209]
 [ 0.09465072  0.10123422  0.10631306  0.20036924  0.10034491]
 [ 0.08010929  0.1104547   0.09665887  0.15891978  0.11896849]
 [ 0.11198978  0.11664006  0.11150688  0.14452107  0.0976636 ]] -> [ 0.10430872]
'''

# train/test split
train_size = int(len(dataY) * 0.7)
# 약 70% data는 training data
test_size = len(dataY) - train_size
# 나머지를 test data
trainX, testX = np.array(dataX[0:train_size]), np.array(
    dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(
    dataY[train_size:len(dataY)])

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
# 앞에 None은 batch_size를 뜻한다.
Y = tf.placeholder(tf.float32, [None, 1])
# 출력값이 1이니까 1로 고정

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
# hideen_dim는 자유롭게 정해주면 된다.
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output
# outputs이 여러군데(각 cell)에서 나올텐데 우리는 마지막꺼[:, -1]만 쓰겠다.
# Y의 예측값이다.

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()