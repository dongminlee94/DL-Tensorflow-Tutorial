import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
print(image.shape)
plt.imshow(image.reshape(3, 3), cmap='Greys')
plt.show()

weight = tf.constant([[[[1.]], [[1.]]], [[[1.]], [[1.]]]])
print(weight.shape)
#
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
# input : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
# filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.
# strides : 크기 4인 1차원 리스트. [0], [3]은 반드시 1. 일반적으로 [1], [2]는 같은 값 사용.
# padding : 'SAME' 또는 'VALID'. 패딩을 추가하는 공식의 차이.
# SAME은 출력 크기를 입력과 같게 유지.

# 3x3x1 필터를 32개 만드는 것을 코드로 표현하면 [3, 3, 1, 32]가 된다.
# 순서대로 너비(3), 높이(3), 입력 채널(1), 출력 채널(32)을 뜻한다. 32개의 출력이 만들어진다.

conv2d_img = conv2d.eval()
# eval로 이미지를 실행시킨다.
print(conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0 , 3)

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1, 2, i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    plt.show()

#####################################################################################33

# 필터를 3개로 적용
weight1 = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]], [[[1., 10., -1.]], [[1., 10., -1.]]]])
print(weight1.shape)

conv2d1 = tf.nn.conv2d(image, weight1, strides=[1,1,1,1], padding='SAME')
conv2d1_img = conv2d1.eval()
print(conv2d1_img.shape)

conv2d1_img = np.swapaxes(conv2d1_img, 0, 3)
for i, one_img in enumerate(conv2d1_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 3, i + 1), plt.imshow(one_img.reshape(3, 3), cmap='gray')
    plt.show()

# 여기까지 Convolution end
#####################################################################33

Max Pooling

image1 = np.array([[[[4], [3]], [[2], [1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image1, ksize=[1, 2, 2, 1], strides=[1,1,1,1], padding='SAME')
print(pool.shape)
print(pool.eval())

#########################################################################

# MNIST image loading

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img = mnist.train.images[0].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

# MNIST Convolution layer
sess = tf.InteractiveSession()

img = img.reshape(-1, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(14, 14), cmap='gray')
    # plt.show()

pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)

for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i + 1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
    plt.show()
