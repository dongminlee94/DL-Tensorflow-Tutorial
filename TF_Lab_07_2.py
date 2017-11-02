# MNIST_dataset

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 이렇게 one_hot을 True로하면 바로 one_hot encoding처리가 된다.

X = tf.placeholder(tf.float32, shape=[None, 784])
# 28 * 28 이니까 784
Y = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

H = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(H, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
# epoch이라는 것은 전체를 두고 1번 돌리는 값
# 여기서는 총 15번을 돌린다고 한다.
# 많을수록 좋다.
batch_size = 100
# 데이터양이 크기때문에 한번에 다 학습시키지 않고 나눠서 학습 시킨다.
# batch_size = 100이 바로 그 몇 개씩 나누는 지 정하는 것

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
            c, _ = sess.run([cost, optimizer], feed_dict={X : batch_xs, Y : batch_ys})
            avg_cost += c / total_batch
        # 여기 for문이 끝나면 1epoch가 끝난다.
        print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X : mnist.test.images, Y : mnist.test.labels}))

#######################################################################

    r = random.randint(0, mnist.test.num_examples - 1)
    # 랜덤한 숫자 하나를 읽어온다.
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    # mnist.test.labels[r:r+1] 이게 one_hot에서 끌고온 숫자가 된다.
    print("Prediction : ", sess.run(tf.argmax(H, 1), feed_dict={X : mnist.test.images[r: r+1]}))
    plt.imshow(mnist.test.images[r : r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()
