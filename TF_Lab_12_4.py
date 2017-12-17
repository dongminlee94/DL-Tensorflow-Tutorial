# Stacked RNN + Softmax Layer

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)

sentence = ("if you want to build a ship, don't drum up people together to"
            "collect wood and don't assign them tasks and work, but rather"
            "teach them to long for the endless immensity of the sea")

# 중복되는 알파벳은 제거

char_set = list(set(sentence))
print(char_set)
'''
['p', 'e', 'u', 'c', 't', 'b', 'i', 'w', 'k', 'f', 'r', 'g', "'", 'l', ' ', 'n', 'd', 'a', ',', 
    'm', 'o', 'h', 'y', 's']
'''
char_dic = {w: i for i, w in enumerate(char_set)}
print(char_dic)
'''
{'p': 0, 'e': 1, 'u': 2, 'c': 3, 't': 4, 'b': 5, 'i': 6, 'w': 7, 'k': 8, 'f': 9, 'r': 10, 
    'g': 11, "'": 12, 'l': 13, ' ': 14, 'n': 15, 'd': 16, 'a': 17, ',': 18, 'm': 19, 'o': 20, 
        'h': 21, 'y': 22, 's': 23}
'''

# hyper parameters
data_dim = len(char_set) # 24
hidden_size = len(char_set)
num_classes = len(char_set) # output size
sequence_length = 10  # Any arbitrary number
# 얼마나 끊어서 읽을 것인가?
learning_rate = 0.1

dataX = []
dataY = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

# training dataset
'''
0 if you wan -> f you want
1 f you want -> you want
2 you want -> you want t
3 you want t -> ou want to
...
168  of the se -> of the sea.
169 of the sea -> f the sea.
'''

batch_size = len(dataX)
# 169개를 위 dataset처럼 나눠서 넣는다.

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# One-hot encoding
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape

# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    # hidden_size는 출력을 얼마로 할 것인가?
    return cell

multi_cells1 = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)
# multi_cells = rnn.MultiRNNCell([lstm_cell()] * 2, state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells1, X_one_hot, dtype=tf.float32)

# FC layer(Softmax)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# Softmax에 들어갈 수 있도록 shape을 조정해준다.
# RNN에서 나온 hidden_size에만 맞도록 나머진 알아서 쌓을 수 있도록 만든다.
# 기계적으로 항상 이런 식으로 한다.
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# 이 코드는 하나의 H를 만들수 있다.
# softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
# softmax_b = tf.get_variable("softmax_b", [num_classes])
# outputs = tf.matmul(X_for_fc, softmax_w) + softmax_b
# 와 같은 코드이다.


# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
# (outputs, 에서 이 outputs은 softmax에서 나온 output이다.
# Softmax에 나와서 다시 펼칠 수 있도록 다시 shape을 바꿔준다.
# Softmax에 맞는 W를 디자인 해주면된다.
# 제일중요!!, RNN's output shape과 동일해야한다.

# All weights are 1 (equal weights)
weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# 위에 있는 outputs를 logits에 넣어주어야 한다.
# 캐중요!, activation_function을 거치지 않은 logit으로 정해주어야한다!!
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        _, l, results = sess.run([train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            print(i, j, ''.join([char_set[t] for t in index]), l)

    # Let's print the last char of each result to check it works
    results = sess.run(outputs, feed_dict={X: dataX})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  # print all for the first result to make a sentence
            print(''.join([char_set[t] for t in index]), end='')
        else:
            print(char_set[index[-1]], end='')
