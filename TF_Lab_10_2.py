# TF_Lab_10_1 tensorboard

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
keep_prob = tf.placeholder(tf.float32)
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.variable_scope("layer1"):
    W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([512]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

with tf.variable_scope("layer2"):
    W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([512]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

with tf.variable_scope("layer3"):
    W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([512]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

with tf.variable_scope("layer4"):
    W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([512]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

with tf.variable_scope("layer5"):
    W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([10]))
    H = tf.matmul(L4, W5) + b5

with tf.variable_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.variable_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(H, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/logs/mnist_relu_r0.001")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += c / total_batch

        print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Accuracy : ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction : ", sess.run(tf.argmax(H, 1), feed_dict={X: mnist.test.images[r: r + 1]}))
