import tensorflow as tf
import numpy as np
# Fancy Sortmax Classifier - cross_entropy, one_hot, reshape

# softmax_cross_entropy_with_logits
# 항상 logits을 생각해야한다.

# 한가지 예를 통해서 학습한다.
# 동물들의 특징들이 나와있고, 마지막은 대략 0~6으로 동물들을 분류할 수 있다.

tf.set_random_seed(777)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Y의 shape에 집중해보자.

nb_classes = 7
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1]) # 0~6, shape=(? , 1)

Y_one_hot = tf.one_hot(Y, nb_classes) # one_hot shape=(?, 1, 7) 여기서 차원이 하나가 늘어난다.
# Y_one_hot은 0~6이라서 7가 된다.
# one_hot이라는 함수를 돌리면 rank N이 rank N+1이 된다!!
# rank가 2차원이면 3차원이 되버린다.
# ex) [[0], [3]]이것이 [[[1000000], [0001000]]]
# 여기서 1000000에서 1은 0을 뜻한다.
# 그래서 reshape이라는 것을 사용한다.
# 차원이 만약에 증가하지 않는다면 다른 element의 one_hot된 결과랑 구분이 안가기 때문에
# 차원을 증가시켜서 구분을 만들어준다.
# ex) [1,2]일때 one_hot된 결과를 한 차원 증가시키지 않는다면 [0,1,0...,0,0,1] 이렇게 된다
# 각 element(각 1, 2)의 결과가 누구인지 구분하기도 힘들고 다루기도 골치아프다.
# 그래서 가장 좋은 방법은 [[0,1,0...], [0,0,1...,.]]으로 만들어주는 것이다.
# 우리가 원하는 것은 0~6까지 숫자가 아니라 [0,1,0,,,] 이러한 숫자들이다.

Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape=(? , 7)
# 여기서 -1은 차원을 얼마나 줄일 것인지를 말한다.
# 뒤에는 7, 앞에는 알아서
# 그래서 여기서 [[1000000], [0001000]]으로 다시 돌아오게 된다.
# 여기가 우리가 원하는 차원, 모양을 맞춰줄 수 있다.

W = tf.Variable(tf.random_normal([16, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
# 여기서 W와 b는 고를 수 있는 숫자를 준다.

logits = tf.matmul(X, W) + b
H = tf.nn.softmax(logits)
# H를 확률적으로 바꾼다. 다 더하면 1로 되도록

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)
# Lab6o1에서 -tf.reduce_sum(Y*tf.log(H), axis=1)이 부분을
# 간결하게 사용할 수 있도록 logits, labels만 정해주면 사용할 수 있도록 해주는 함수이다.
# reduce_sum까지만 해주는 함수.

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#########################x####3
# 학습을 시켜보자.

prediction = tf.argmax(H, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# Y_one_hot은 레이블이 된다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X : x_data, Y : y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X : x_data, Y : y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
            # format이라는 함수는 저 {}안에 있는 공간에 ()에 있는 것을 채워넣는다.

    pred = sess.run(prediction, feed_dict={X : x_data})

    for p, y in zip(pred, y_data.flatten()):
        # zip이라고 하는 것은 List여러개를 slice할 때 사용한다.
        # [[1],[0]] -> [1, 0]해주는것이 flatten이다.
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

##################################################################################

# code practice 1

tf.set_random_seed(777)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb = 7
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb)
Y_one_hot = tf.reshape(Y_one_hot,[-1, nb])

W = tf.Variable(tf.random_normal([16, nb]))
b = tf.Variable(tf.random_normal([nb]))

logits = tf.matmul(X, W) + b
H = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(H, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X : x_data, Y : y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X : x_data, Y : y_data})
            print(step, loss, acc)

    pred = sess.run(prediction, feed_dict={X : x_data})

    for p, y in zip(pred, y_data.flatten()):
        print(p == int(y), p, int(y))

###################################################################################

# code practice 2

tf.set_random_seed(777)

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.int32, shape=[None, 1])
# 0~6, shape=(?, 1)

Y_one_hot = tf.one_hot(Y, 7)
# 0부터 6까지 라벨링먼저, shape=(?, 1, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])
# 차원하나 늘어나서 우리가 원하는 shape모양으로 바꿔주기
# -1은 everything
# shape=(?, 7)

W = tf.Variable(tf.random_normal([16, 7]))
b = tf.Variable(tf.random_normal([7]))

logits = tf.matmul(X, W) + b
H = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.argmax(H, 1)
correct_prediction = tf.equal(predicted, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        sess.run(train, feed_dict={X : x_data, Y : y_data})
        if step % 200 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X : x_data, Y : y_data})
            print(step, loss, acc)

    pred = sess.run(predicted, feed_dict={X : x_data})

    for p, y in zip(pred, y_data.flatten()):
        print(p == int(y), p, int(y))
