import tensorflow as tf

# multi-variable linear regression(그냥 쌩으로 구하기)
x1_data = [73, 93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]), name='weigt1')
w2 = tf.Variable(tf.random_normal([1]), name='weigt2')
w3 = tf.Variable(tf.random_normal([1]), name='weigt3')
b = tf.Variable(tf.random_normal([1]), name='bias')

H = x1 * w1 + x2 * w2 + x3 * w3 + b

cost = tf.reduce_mean(tf.square(H - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# 1e-5승으로도 둘 수 있다.
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost :", cost_val, "\nPridiction:\n", H_val)
        #cost값은 점점 0으로 수렴하고, Pridiction은 점점 Y로 수렴하는 것으로 볼 수가 있다.

###########################################################################

multi-variable linear regression(matrix로 구하기)

x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
y_data = [[152], [185], [180], [196], [142]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])
# shape을 정확하게 입력 해야한다!
# shape의 앞에꺼는 instance의 개수
# shape의 뒤에꺼는 instance안에 있는 data의 개수
# None이라고 하는 것은 니가 원하는 만큼 데이터를 둘 수 있다. N개

W = tf.Variable(tf.random_normal([3, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

H = tf.matmul(X, W) + b
# matmul은 matrix 곱하기하는 것

cost = tf.reduce_mean(tf.square(H - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_data, Y: y_data})
    if step % 10 == 0:
        print(step, cost_val, H_val)

################################################################################################

# code practice

x_data = [[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]]
y_data = [[152], [185], [180], [196], [142]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal(shape=[3, 1]))
b = tf.Variable(tf.random_normal(shape=[1]))

H = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(H, feed_dict={X: [[73, 80, 90], [93, 88, 77], [89, 91, 100], [96, 98, 95], [73, 66, 80]]}))
