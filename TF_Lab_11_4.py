import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100

class Modeling:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.__build_net()

    def __build_net(self):
        with tf.variable_scope(self.name):
            self.training = tf.placeholder(tf.bool)

            self.X = tf.placeholder(tf.float32, [None, 784])
            X_img = tf.reshape(self.X, [-1, 28, 28, 1])
            self.Y = tf.placeholder(tf.float32, [None, 10])

            with tf.variable_scope("layer1"):
                conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3],
                                         padding='SAME', activation=tf.nn.relu)
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2],
                                                padding='SAME', strides=2)
                dropout1 = tf.layers.dropout(inputs=pool1, rate=0.7, training=self.training)

            with tf.variable_scope("layer2"):
                conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3],
                                         padding='SAME', activation=tf.nn.relu)
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2],
                                                padding='SAME', strides=2)
                dropout2 = tf.layers.dropout(inputs=pool2, rate=0.7, training=self.training)

            with tf.variable_scope("layer3"):
                conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3],
                                         padding='SAME', activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2],
                                                padding='SAME', strides=2)
                dropout3 = tf.layers.dropout(inputs=pool3, rate=0.7, training=self.training)

            with tf.variable_scope("dense"):
                flat = tf.reshape(dropout3, [-1, 128 * 4 * 4])
                dense = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(inputs=dense, rate=0.5, training=self.training)

                self.logits = tf.layers.dense(inputs=dropout4, units=10)

            with tf.variable_scope("cost"):
                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.Y))
                self.cost_summ = tf.summary.scalar("cost", self.cost)

            with tf.variable_scope("optimizer"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

            self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.X: x_test,
                                                     self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test,
                                                       self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data,
                self.Y: y_data, self.training: training})

with tf.Session() as sess:
    models = []
    num_models = 7

    for m in range(num_models):
        models.append(Modeling(sess, "model" + str(m)))

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/logs/cnn_mnist_r0.001")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    print('학습을 시작해볼까? 바로 ㄱㄱ')

    # train my model
    for epoch in range(training_epochs):
        avg_cost_list = np.zeros(len(models))
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            for m_index, m in enumerate(models):
                # enumerate는 "열거하다"라는 뜻으로서, 이 함수는 순서가 있는 자료형
                # (리스트, 튜플, 문자열)을 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다.
                '''for i, name in enumerate(['body', 'foo', 'bar'])
                    print(i, name)
                0 body
                1 foo
                2 bar'''
                c, _ = m.train(batch_xs, batch_ys)
                avg_cost_list[m_index] += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

    print('Learning Finished!')

    # Test model and check accuracy
    test_size = len(mnist.test.labels)
    predictions = np.zeros(test_size * 10).reshape(test_size, 10)

    for m_index, m in enumerate(models):
        print(m_index, 'Accuracy', m.get_accuracy(mnist.test.images, mnist.test.labels))
        p = m.predict(mnist.test.images)
        predictions += p

    ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(mnist.test.labels, 1))
    ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
    print('Ensemble accurcracy:', sess.run(ensemble_accuracy))

