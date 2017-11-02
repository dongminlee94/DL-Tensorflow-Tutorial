# TF로 간단한 Linear regression 구현

import tensorflow as tf

# H(x) = Wx + b

# 두개의 데이터
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# 값을 지정하지 말고 Lab1에서 배운 Placeholders 이용하기
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
# 여기서 shape도 내 마음대로 조정할 수 있다.
# placeholder를 사용하는 이유는 L.R 모델을 만들고 내 마음대로 입력할 수 있기 때문이다.

W = tf.Variable(tf.random_normal([1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
# 텐서플로가 사용하는 Variable
# 변수 x, 우리가 사용하는 것이 아니라 텐서플로가 사용하는 것
# 텐서플로가 자체적으로 변경되는 값
# [1]는 shape을 정의한 것, rank가 1인 것

# 함수만들기
# hypothesis = x_train * W + b

# Placeholder를 이용한 함수
hypothesis = X * W + b

# cost(W,b)
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Placeholder를 이용한 cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# square는 제곱
# reduce_mean은 평균을 내주는 것

# cost를 minimize하는 것
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# root텐서의 이름 train, cost를 minimize한 것, cost는 가설-y
# "경사타고 내려가기", 미분을 통해 최저 비용을 향해 진행하도록 만드는 핵심함수!
# 전부 다 이어져있다

# 실행하기
sess = tf.Session()
# global_variables로 초기설정
sess.run(tf.global_variables_initializer())

# Placeholder 이용한 for문
for step in range(1501):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : x_train, Y : y_train})
    # sess.run([ 여기서 묶어서 한번에 실행시킬 수 있다. ])
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# 추가 for문
# for step in range(2001):
#     cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1, 2, 3, 4, 5], Y : [2.1, 3.1, 4.1, 5.1, 6.1]})
#     # sess.run([ 여기서 묶어서 한번에 실행시킬 수 있다. ])
#     if step % 20 == 0:
#         print(step, cost_val, W_val, b_val)
#         # W가 1, b가 1.1에 가까워진다.

print(sess.run(hypothesis, feed_dict={X : [5]}))
print(sess.run(hypothesis, feed_dict={X : [2.5]}))
print(sess.run(hypothesis, feed_dict={X : [1.5, 3.5]}))

###############################################################################################

# code practice 1

# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]), name='Weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# H = x_train * W + b
H = X * W + b

# cost = tf.reduce_mean(tf.square(H - y_train))
cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for step in range(1501):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

for step in range(1501):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1, 2, 3], Y : [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(H, feed_dict={X : [5]}))
print(sess.run(H, feed_dict={X : [2, 5]}))
print(sess.run(H, feed_dict={X : [1.5, 3.5]}))

#####################################################################################################33

# code practice 2

x = [1,2,3]
y = [1,2,3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

# H = x * W + b
H = X * W + b
# cost = tf.reduce_mean(tf.square(H - y))
cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for step in range(1501):
#     sess.run(train)
#     if step % 50 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

for step in range(1501):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1,2,3,], Y : [1,2,3]})
    if step % 50 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(H, feed_dict={X : [5]}))
print(sess.run(H, feed_dict={X : [2, 5]}))
print(sess.run(H, feed_dict={X : [1.5, 3.5]}))

#####################################################################################################33

code practice 3

# X = [1,2,3]
# Y = [1,2,3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

H = X * W + b
cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# for step in range(1501):
#     sess.run(train)
#     if step % 50 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))

for step in range(1501):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1,2,3], Y : [1,2,3]})
    if step % 50 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(H, feed_dict={X : [5]}))
print(sess.run(H, feed_dict={X : [2, 5]}))
print(sess.run(H, feed_dict={X : [1.5, 3.5]}))

#####################################################################################################33

code practice 4

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

H = X * W + b
cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1501):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1,2,3], Y : [1,2,3]})
    if step % 50 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(H, feed_dict={X : [1,4,8]}))

#####################################################################################################33

code practice 5

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

H = X * W + b
cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1,2,3], Y : [1,2,3]})
    if step % 50 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(H, feed_dict={X : [1,2,3]}))

#####################################################################################################33

code practice 6

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

H = X * W + b
cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X : [1,2,3], Y : [1,2,3]})
        if step % 50 == 0:
            print(step, cost_val, W_val, b_val)

    print(sess.run(H, feed_dict={X : [1,4,7]}))

#####################################################################################################33

# code practice 7

# Learning data
x_data = [1,2,3]
y_data = [1,2,3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
H = X * W + b
cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Learning
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X : x_data, Y : y_data})
        if step % 250 == 0:
            print(step, cost_val)

    # Test
    print(sess.run(H, feed_dict={X : [5]}))
    print(sess.run(H, feed_dict={X : [2.5]}))
    print(sess.run(H, feed_dict={X : [1.5, 3, 4.5]}))
