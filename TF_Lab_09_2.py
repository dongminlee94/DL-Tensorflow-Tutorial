import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

with tf.name_scope("layer1") as scope1:
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    # w1_hist = tf.summary.histogram("weight1", W1)
    # b1_hist = tf.summary.histogram("biases1", b1)
    # layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope2:
    W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    H = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    # w2_hist = tf.summary.histogram("weight2", W2)
    # b2_hist = tf.summary.histogram("biases2", b2)
    # layer2_hist = tf.summary.histogram("H", H)

with tf.name_scope("cost") as scope3:
    cost = -tf.reduce_mean(Y*tf.log(H) + (1-Y)*tf.log(1-H))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope4:
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.cast(H > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_sum = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/logs/xor_logs_r0.1")
    writer.add_graph(sess.graph)
    # summary마다 업데이트를 계속 하겠다.
    # 내가 몇번의 한번씩 merge를 하겠다.
    # 매 회마다 기록을 하겠다.
    # add_summary가 name_scope내용을 계속 업데이트 하는 것, 적어나가는 것

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X : x_data, Y : y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run([W1, W2]))

    h, c, a = sess.run([H, predicted, accuracy], feed_dict={X : x_data, Y : y_data})
    print(h, c, a)
