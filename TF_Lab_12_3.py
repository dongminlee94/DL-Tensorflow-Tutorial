# Long Sequence RNN

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sample = "if you want you"
# string 문자열을 하나 둔다.

idx2char = list(set(sample))
# set을 주면 유니크한 문자열이 만들어지고
# list를 쓰면 그것이 밑에 보이는 것과 같이 list형식으로 출력이 이루어진다.

# 계속 랜덤으로 재생할 것이다.

# idx2char output test
print(idx2char)
'''
['o', 'f', 'u', 'y', 'w', 't', 'a', 'n', ' ', 'i']
'''

char2idx = {c: i for i, c in enumerate(idx2char)}
# 문자열을 주면 char형으로 바꾼 것에 번호를 매긴다.

# char2idx output test
print(char2idx)
'''
{'o': 0, 'f': 1, 'u': 2, 'y': 3, 'w': 4, 't': 5, 'a': 6, 'n': 7, ' ': 8, 'i': 9}
'''

# hyper parameters
# hyper parameters을 정해주는 이유는 input이 아무 단어가 와도 알아서 정해주도록 만들기 위해서이다.

dic_size = len(char2idx)  # RNN 'input' size (one hot size)
# 10, 문장전체에서 쓰일 단어(입력될 글자)의 size
# 'o': 0, 'f': 1, 'u': 2, 'y': 3, 'w': 4, 't': 5, 'a': 6, 'n': 7, ' ': 8, 'i': 9

hidden_size = len(char2idx)  # RNN 'output' size
# 10, RNN하고 나서 출력의 size
# Cell에서 통과하고 나서 one_hot으로 했을 때 몇 size로 출력할꺼야?

num_classes = len(char2idx)  # final 'output' size (RNN or 'softmax', etc.)
# 10, 출력의 class의 size, 최종적으로 나올 size, 알파벳의 개수

batch_size = 1  # one sample data, one batch
# 1, 한 번 최종단어가 나올 때 Yt가 나올때 이걸 몇번으로 끊어서 할 것인가.
# 문장나누기. Enter역할

sequence_length = len(sample) - 1  # number of LSTM rollings
# 14, 한번에 단어를 몇개 넣을 것인가? or 셀을 몇 개 둘 것인가?
# 14개니까 처음에 ex) 'if you want yo'까지
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]
# char2dix으로 index를 지정한 것으로 sample에 있는 것들을 번호를 매긴다.

# sample_idx output test
print(sample_idx)
'''
[9, 1, 8, 3, 0, 2, 8, 4, 6, 7, 5, 8, 3, 0, 2]
'''

x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]
# x_data는 처음부터 마지막전까지
# y_data는 두번째부터 마지막까지
# 1차원이라 행만 있음.

# ex) "if you want you"라면
# x_data는 if you want yo까지
# y_data는 f you want you이다.

# x_data, y_data output test
print(x_data)
print(y_data)
'''
[[9, 1, 8, 3, 0, 2, 8, 4, 6, 7, 5, 8, 3, 0]]
[[1, 8, 3, 0, 2, 8, 4, 6, 7, 5, 8, 3, 0, 2]]
'''

X = tf.placeholder(tf.int32, [None, sequence_length]) # X data
Y = tf.placeholder(tf.int32, [None, sequence_length]) # Y label

x_one_hot = tf.one_hot(X, num_classes)
# one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
# ont hot함수를 사용하면 알아서 one hot처리를 해준다.

# one hot을 만들 때 주의점
# dim에 주의하라.
# shape이 어떻게 변하는지 잘 살펴봐야한다.

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
# cell이 input
# x_one_hot이 target 각 H1, H2에서 나올 값

# FC layer(Softmax)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(300):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))


