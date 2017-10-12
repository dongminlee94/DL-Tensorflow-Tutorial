import tensorflow as tf

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]
# y_data에는 One-Hot encoding 인코딩 방법을 사용한다.
# 하나만 핫하다.
# 세자리 0 를 만들어 주고, 각각 그 위치가 제일 핫하다는 것을 보여준다.
#        1
#        2
# 0이면 [1,0,0], 1이면 [0,1,0], 2이면 [0,0,1]가 된다.

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])
# 3이라는 것은 레이블의 개수(즉, 0이냐 1이냐 2냐 하는 그 레이블의 개수)
# Y = tf.placeholder("float", [None, 3])으로 해도된다.

W = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))

H = tf.nn.softmax(tf.matmul(X, W) + b)
# softmax 라이브러리를 쓰면 softmax function을 사용하여 확률적으로 바꿔서 알려준다.
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H), axis=1))
# reduce_sum은 sigma처럼 더해주는 것
# 여기서의 cost는 cross-entropy cost를 쓴 것이다.
# axis는 차원(축)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X : x_data, Y : y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}))

    # 실습1
    a = sess.run(H, feed_dict={X : [[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1)))
    # arg_max a중에서 어느 것이 가장 높은지 물어보는 함수이다.
    # 여기서 1은 학점으로 치면 b를 맞는다.
    # 뒤에 1은 차원 dimenssion을 말한다.

    # 실습2
    all = sess.run(H, feed_dict={X : [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))
    # 여기서 all은 학점으로 치면 b a c이다.

#################################################################################

# code practice 1

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4], [4,1,5,5], [1,7,5,5], [1,2,5,6], [1,6,6,6], [1,7,7,7]]
y_data = [[0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [1,0,0], [1,0,0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])

W = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))

H = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H), axis=1))
# y가 1일때만 한다.
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train, feed_dict={X : x_data, Y : y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}))

    a = sess.run(H, feed_dict={X : [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))
