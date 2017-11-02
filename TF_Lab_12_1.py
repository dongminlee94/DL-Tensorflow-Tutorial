# RNN Basics

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

with tf.variable_scope('one_cell'):
    # One cell RNN input_dim (4) -> output_dim (2)
    hidden_size = 2 # 출력값 개수
    cell = rnn.BasicRNNCell(num_units=hidden_size)
    # cell = rnn.BasicLSTMCell(num_units=hidden_size)
    print(cell.output_size, cell.state_size)

    x_data = np.array([[h]], dtype=np.float32) # h = [1,0,0,0]
    pp.pprint(x_data)
    # pprint(pretty printer) : 쉽고 예쁘게 출력하기
    # print(x_data)
    outputs, states0 = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    # print(outputs.eval())
    pp.pprint(outputs.eval())
    # output
    # array([[[0.47614652, -0.07847963]]], dtype=float32)
    # W의 초깃값에 대해서 출력물이 나왔다.

with tf.variable_scope('two_sequances'):
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)

    x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
    # 위에 h, e, l, o를 각각 one_hot_encoding을 해주었기 때문에
    print(x_data.shape)
    pp.pprint(x_data)
    outputs, states1 = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

with tf.variable_scope('3_batches'):
    # One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

    hidden_size = 2
    cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, states2 = tf.nn.dynamic_rnn(
        cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())


