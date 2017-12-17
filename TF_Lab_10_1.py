# NN, ReLu, Xavier, dropout, and AdamOptimizer

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
keep_prob = tf.placeholder(tf.float32)
# dropout (keep_prob) rate 0.7 on training, but should be 1 for testing
# 밑에서 0.7로 트레이닝을 시키고, 테스팅할때는 1로 한다.
# Network의 학습, 테스팅에 얼마나 유지, 학습에 사용할 지 설정값을 placeholder 변수로 초기화한다.
# Train과 Testing 두 곳에서 각각 다른 값을 사용해야하기 때문에..
training_epochs = 15
# epoch이라는 것은 전체를 두고 1번 돌리는 값
# 여기서는 총 15번을 돌린다고 한다.
# 많을수록 좋다.
batch_size = 100
# 데이터양이 크기때문에 한번에 다 학습시키지 않고 나눠서 학습 시킨다.
# batch_size = 100이 바로 그 몇 개씩 나누는 지 정하는 것

# input place holders
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
H = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(H, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        # 몇번 epoch을 할 것인가??
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        # epoch 하나당 batch_size 1번 돈다.
        # ex) 10000인데 batch_size가 100이면 100번을 반복(iteration)해야 1epoch가 된다.
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X : batch_xs, Y : batch_ys, keep_prob: 0.7})
            # keep_prob: 0.7은 70%로 적용시킨다는 뜻
            avg_cost += c / total_batch
        # 여기 for문이 끝나면 1epoch가 끝난다.
        print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={X : mnist.test.images, Y : mnist.test.labels, keep_prob : 1}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    # 랜덤한 숫자 하나를 읽어온다.
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    # mnist.test.labels[r:r+1] 이게 one_hot에서 끌고온 숫자가 된다.
    print("Prediction : ", sess.run(tf.argmax(H, 1), feed_dict={X : mnist.test.images[r: r+1]}))
    plt.imshow(mnist.test.images[r : r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
