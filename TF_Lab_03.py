import tensorflow as tf
import matplotlib.pyplot as plt
# 그래프를 그릴 때 사용하는 라이브러리

# H(x) = Wx, b는 여기서 없다.

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

H = X * W

cost = tf.reduce_mean(tf.square(H - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []
# 그래프를 그리기 위해서 저장할 리스트를 만든다.

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W : feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()
# cost를 시각화 하기 위해서 만든 것
# 우리가 Linear Regression 한 것을 시각화 해본 것

###########################################################

# x_data = [1, 2, 3]
# y_data = [1, 2, 3]

# test1
X = [1, 2, 3]
Y = [1, 2, 3]

# W = tf.Variable(tf.random_normal([1]), name='Weight')

# test2
# 엄청 먼 값으로 test 해볼려고 한다.
# W = tf.Variable(5.0)
W = tf.Variable(-3.0)

# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)

H = X * W

cost = tf.reduce_sum(tf.square(H - Y))

# # Gradient descent algorithm 적용
# # 식 그대로 적용!
# learning_rate = 0.1
# # 알파의 값을 기본적으로 0.1로 둔다.
# gradient = tf.reduce_mean((W * X - Y) * X)
# descent = W - learning_rate * gradient
# # 이게 새로운 W의 값
# update = W.assign(descent)
# # assign하기
# # update 매우 중요

# 하지만 텐서플로우가 알아서 미분도 해준다 이렇게 쓰면
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
# 개이득

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for step in range(21):
#     sess.run(train, feed_dict={X : x_data, Y : y_data})
#     print(step, sess.run(cost, feed_dict={X : x_data, Y : y_data}), sess.run(W))

# test
for step in range(100):
    print(step, sess.run(W))
    sess.run(train)

#############################################################################################333

# code practice 1

X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)
H = X * W
cost = tf.reduce_mean(tf.square(H - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    cur_cost, cur_W = sess.run([cost, W], feed_dict={W : feed_W})
    W_val.append(cur_W)
    cost_val.append(cur_cost)

plt.plot(W_val, cost_val)
plt.show()
#
# #######################################################################
#
X = [1,2,3]
Y = [1,2,3]

# W = tf.Variable(tf.random_normal([1]))
W = tf.Variable(5.0)
H = X * W
cost = tf.reduce_mean(tf.square(H - Y))

# case1
learning_rate = 0.01
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# case2
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# # case1
# for i in range(21):
#     sess.run(update)
#     print(i, sess.run(cost), sess.run(W))

# case 2
for i in range(100):
    print(i, sess.run(W))
    sess.run(train)

#############################################################################################333

# code practice 2
# 그래프 그리는 코드만 연습

X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)
H = X * W
cost = tf.reduce_mean(tf.square(H - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
W_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W : feed_W})
    # feed_dict로 쓰려면 위에 그 변수가 placeholder로 되어 있어야 한다.
    W_val.append(curr_W)
    cost_val.append(curr_cost)

plt.plot(W_val, cost_val)
plt.show()

#############################################################################################333

# code practice 3

X = [1,2,3]
Y = [1,2,3]

W = tf.placeholder(tf.float32)
H = X * W
cost = tf.reduce_mean(tf.square(H - Y))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost_val = []
    W_val = []

    for step in range(-30, 50):
        feed_W = step * 0.1
        curr_cost, curr_W = sess.run([cost, W], feed_dict={W : feed_W})
        cost_val.append(curr_cost)
        W_val.append(curr_W)

    plt.plot(W_val, cost_val)
    plt.show()
