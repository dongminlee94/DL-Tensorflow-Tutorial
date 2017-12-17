# TF로 파일에서 데이터 읽어오기

import tensorflow as tf
import numpy as np

# numpy는 C 언어에 있는 배열과 같은 형태로 움직이는 '다차원 배열'을 기반으로 하는 모듈이다.

# # Loading data from file
tf.set_random_seed(777)
# seed(보통 시간을 사용하고 좀 더 정확한 난수 값 생성하기 위해 다양한 방법을
# 사용합니다만 범위를 벗어납니다)를 사용하게 되는데 예제에서는 예제이기 때문에
# 항상 동일한 값이 나도록 777로 고정한 것이다.
# seed가 같으면 이후의 랜덤하게 생성되는 숫자들이 동일하게 만들어진다.

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
# csv파일을 가져오는 것
# dlimiter 구획 문자, 구분 문자
x_data = xy[:, 0:-1]
# x데이터의 행은 전부 다, 열은 처음부터 -1 전까지, 즉 y데이터 전까지 읽는다.
y_data = xy[:, [-1]]
# 행은 다 가져오고, 열은 마지막 열만 가져온다.
# y데이터를 출력한다. []리스트화를 통해서 읽는다.

print(x_data.shape, x_data, len(x_data))
# len은 instance의 개수
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
# b는 Y의 개수에 맞게 shape을 설정한다.

H = tf.matmul(X,W) + b
cost = tf.reduce_mean(tf.square(H - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val_1 = []
H_val_1 = []

for step in range(2001):
    cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_data, Y : y_data})
    cost_val_1.append(cost_val)
    H_val_1.append(H_val)
    if step % 50 == 0:
        print(step, cost_val, H_val)

print("Your score will be ", sess.run(H, feed_dict={X : [[100, 70, 90]]}))
print("Other score will be ", sess.run(H, feed_dict={X : [[60, 70, 80], [90, 90, 80]]}))

#################################################################

b = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
1 2 3 4
5 6 7 8
9 10 11 12

matrix 설명
b[숫자, 숫자], b[숫자:숫자, 숫자:숫자] 형식으로 나타낼 수 있다.
, 앞에는 행 / , 뒤에는 열을 표시한다.
행과 열은 0부터 시작한다.

print(b[:, 1])
[2, 6, 10]
1열을 출력한다.

print(b[-1])
print(b[-1, :])
[9, 10, 11, 12]
마지막 행을 출력한다.

print(b[-2, :])
[5, 6, 7, 8]
뒤에서 두번째 행을 출력한다.

print(b[-1, ...])
:은 ...으로 대체할 수 있다.

print(b[0:2, :])
print(b[0:2])
[[1 2 3 4] [5 6 7 8]]
행을 출력할 때는 :가 굳이 필요한건 아니다.

#############################################################

# 텐서플로 책보고 따라한 것.

n_points = 1000
vectors_set = []

for i in range(n_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.plot(x_data, y_data, 'ro')
plt.show()

############################################################33

# 파일을 과연 어떻게 읽어 올까??
tf.set_random_seed(777)

filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], shuffle=False)

reader = tf.TextLineReader()
# 파일을 읽어 올 reader를 정의한다.
# 파일에서 데이터를 읽어오는 컴포넌트를 Reader라고 한다.
# 미리 정의된 reader를 쓰면 더 편리하다.

key, value = reader.read(filename_queue)
# key와 value로 나눠서 읽겠습니다.
# 텍스트 파일을 읽을 때 기본적으로 많이 사용한다.

# ex) csv파일에 아래와 같은 문자열이 들어가 있다고 할 때
#
# 167c9599-c97d-4d42-bdb1-027ddaed07c0,1,2016,REG,3:54
# 67ea7e52-333e-43f3-a668-6d7893baa8fb,1,2016,REG,2:11
# 9e44593b-a870-446e-aed5-90a22ab0c952,1,2016,REG,2:32
# 48832a52-e56c-467f-a1ef-c6f8c6e908ea,1,2016,REG,2:17
#
# 위의 코드 처럼, TextLineReader를 이용하여 파일을 읽게 되면 value에
# 처음에는 “167c9599-c97d-4d42-bdb1-027ddaed07c0,1,2016,REG,3:54”이,
# 다음에는 “67ea7e52-333e-43f3-a668-6d7893baa8fb,1,2016,REG,2:11” 문자열이
# 순차적으로 리턴된다.

# Reader라는 것은 그저 읽는 것. decoder는 안한다.

record_defaults = [[0], [0], [0], [0]]
# 파싱할 데이터타입은 이렇게 생겼다. 라는 것을 보여준다.
xy = tf.decode_csv(value, record_defaults=record_defaults)
# reader에서 읽은 값은 파일의 원시 데이터이다. 아직 파싱(해석)이 된 데이터가 아니다.
# decode_csv로 decode를 해서

# 즉 우리가 학습에서 사용할 데이타는
# 167c9599-c97d-4d42-bdb1-027ddaed07c0,1,2016,REG,3:54
# 하나의 문자열이 아니라
# Id = “167c9599-c97d-4d42-bdb1-027ddaed07c0”,
# Num  = 1
# Year = 2016
# rType = “REG”
# rTime = “3:54”
# 과 같이 문자열이 파싱된 각 필드의 값이 필요하다.
# 이렇게 읽어드린 데이터를 파싱(해석)하는 컴포넌트를 decoder라고 한다.

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
# 행의 개수가 이미 정해졌기에 열만 인덱싱한다.
# batch는 폼프와 같은 역할, 그 데이터에서 축으로 읽어올 수 있도록 해준다.
# 말 그대로 이렇게 배치를 해서 읽겠다.
# batch로 묶고자 하는 tensor들을 정의한다.
# 앞에꺼는 x_data, 뒤에꺼는 y_data이다.
# batch_size라는 것은 한번에 몇개를 가져올 것인지를 정해준다. "10개씩 가져와라"

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
H = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(H - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# 이부분은 우리가 일반적으로 쓰는 부분이다.

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_batch, Y : y_batch})
    if step % 10 == 0:
        print(step, cost_val, H_val)

coord.request_stop()
coord.join(threads)
# 이부분은 우리가 일반적으로 쓰는 부분이다.

############################################################################################33

# code practice 1

tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[: ,0:-1]
y_data = xy[: ,[-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder("float", shape=[None, 3])
Y = tf.placeholder("float", shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost_val1 = []
    H_val1 = []

    for step in range(2001):
        cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_data, Y : y_data})
        cost_val1.append(cost_val)
        H_val1.append(H_val)
        if step % 50 == 0:
            print(step, cost_val, H_val)

    print("나" ,sess.run(H, feed_dict={X : [[100, 70, 90]]}))
    print("너네" ,sess.run(H, feed_dict={X : [[100, 70, 90], [90, 70, 80], [70, 80, 80]]}))

############################################################################################33

# code practice 2

tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 1])

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

H = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost_val1 = []
    H_val1 = []

    for step in range(2001):
        cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_data, Y : y_data})
        cost_val1.append(cost_val)
        H_val1.append(H_val)

    print(sess.run(H, feed_dict={X : [[70, 80, 80]]}))
    print(sess.run(H, feed_dict={X : [[70, 80, 80], [90, 100, 100], [80, 75, 95]]}))

############################################################################################33

# code practice 3

tf.set_random_seed(777)

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
H = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost_val1 = []
    H_val1 = []

    for step in range(2001):
        cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_data, Y : y_data})
        cost_val1.append(cost_val)
        H_val1.append(H_val)

        if step % 100 == 0:
            print(step, cost_val, H_val)

    print("나", sess.run(H, feed_dict={X : [[100, 70, 80]]}))
    print("너네들", sess.run(H, feed_dict={X : [[100, 70, 90], [90, 70, 80], [70, 80, 80]]}))

############################################################################################33

# code practice 4

tf.set_random_seed(777)

xy = np.loadtxt('data-01-test.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))
H = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(H - Y))
train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    cost_val1 = []
    H_val1 = []

    for step in range(2001):
        cost_val, H_val, _ = sess.run([cost, H, train], feed_dict={X : x_data, Y : y_data})
        cost_val1.append(cost_val)
        H_val1.append(H_val)

        if step % 100 == 0:
            print(step, cost_val, H_val)

    print("나", sess.run(H, feed_dict={X: [[100, 70, 80]]}))
    print("너네들", sess.run(H, feed_dict={X: [[100, 70, 90], [90, 70, 80], [70, 80, 80]]}))
