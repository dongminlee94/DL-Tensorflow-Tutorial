import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope("layer1") as scope1:
    W1 = tf.Variable(tf.random_normal([784, 512]), name='weight1')
    b1 = tf.Variable(tf.random_normal([512]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer2") as scope2:
    W2 = tf.Variable(tf.random_normal([512, 512]), name='weight2')
    b2 = tf.Variable(tf.random_normal([512]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

with tf.name_scope("layer3") as scope3:
    W3 = tf.Variable(tf.random_normal([512, 512]), name='weight3')
    b3 = tf.Variable(tf.random_normal([512]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

with tf.name_scope("layer4") as scope4:
    W4 = tf.Variable(tf.random_normal([512, 512]), name='weight4')
    b4 = tf.Variable(tf.random_normal([512]), name='bias4')
    layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

with tf.name_scope("layer5") as scope5:
    W5 = tf.Variable(tf.random_normal([512, 10]), name='weight5')
    b5 = tf.Variable(tf.random_normal([10]), name='bias5')
    H = tf.matmul(layer4, W5) + b5

with tf.name_scope("cost") as cost1:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("optimizer") as optimizer1:
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/logs/mnist_exercise_r0.01")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, c, _ = sess.run([merged_summary, cost, optimizer], feed_dict={X : batch_xs, Y : batch_ys})
            writer.add_summary(summary, global_step=i)
            avg_cost += c / total_batch

        print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy : ", sess.run(accuracy, feed_dict={X : mnist.test.images, Y : mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction : ", sess.run(tf.argmax(H, 1), feed_dict={X : mnist.test.images[r: r+1]}))
