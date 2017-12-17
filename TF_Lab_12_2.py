# RNN 'Hi Hello' training

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # reproducibility

idx2char = ['h', 'i', 'e', 'l', 'o']
# 일종의 딕셔너리를 만들자.
# 0 h
# 1 i
# 2 e
# 3 l
# 4 o

# Teach hello: hihell -> ihello

x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
# 0, 1, 0, 2, 3, 3을 one_hot_encoding을 참고해서 보자.

x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [[1, 0, 2, 3, 3, 4]]    # ihello
# 우리가 output으로 보이고 싶은 데이터 'ihello' 이다.

# RNN parameters
num_classes = 5
# 총 쓰일 알파벳은 5개
input_dim = 5  # one-hot size
hidden_size = 5  # output from the LSTM. 5 to directly predict one-hot!!!
# one_hot과 동일하니까 5로 표현
batch_size = 1   # one sentence, 한 문장이니까.
sequence_length = 6  # 'ihello' == 6
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
# 3차원의 공간으로 만든다.
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# 출력값이니까 hidden_size = 5를 표현한다.

initial_state = cell.zero_state(batch_size, tf.float32)
# 이니셜 state는 다 0으로 준다.(batch_size를 주면 된다.)

outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(
    inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)

# reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
# sequence_loss라는 함수가 짱임.
# logits은 우리의 예측(prediction, targets은 true 데이터 = Y, weights은 다 1, [1, 1, 1]
cost= tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(outputs, axis=2)
# 0, 2, 3이란 예측을 하게 된다.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        l, _ = sess.run([cost, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))

'''
0 loss: 1.71584 prediction:  [[2 2 2 3 3 2]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  eeelle
1 loss: 1.56447 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
2 loss: 1.46284 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
3 loss: 1.38073 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
4 loss: 1.30603 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
5 loss: 1.21498 prediction:  [[3 3 3 3 3 3]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  llllll
6 loss: 1.1029 prediction:  [[3 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  lhlllo
7 loss: 0.982386 prediction:  [[1 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihlllo
8 loss: 0.871259 prediction:  [[1 0 3 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihlllo
9 loss: 0.774338 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
10 loss: 0.676005 prediction:  [[1 0 2 3 3 4]] true Y:  [[1, 0, 2, 3, 3, 4]]
	Prediction str:  ihello
'''