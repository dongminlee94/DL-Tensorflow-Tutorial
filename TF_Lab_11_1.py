# TF CNN Basics

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
# sess = tf.Session()으로 열면 앞부분에 그래프를 먼저 구성하고(모델링)하고
# 세션범위 안에서 값을 sess.un(Tesor)해서 확인해야하지만,
# InteractiveSession의 경우는 현재 세션을 default session으로 인식해 값 확인이 편리하다.
# Ipython(jupyter) notebook 같은 곳에서 InteractiveSession을 사용하면 편리하다.

# tf.Session()을 with와 동시에 사용하는 것과 동일.

# InteractiveSession은 현재 열린 세션을 default 세션으로 인식하기 때문에
# sess.run()을 통해 결과값을 얻지 않아도 되고 Tensor.eval()로 바로 값을 구하거나
# Operation.run()을 통해 함수를 돌릴 수 있습니다.

image = np.array([[[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]], dtype=np.float32)
print(image.shape)
plt.imshow(image.reshape(3, 3), cmap='Greys')
plt.show()
# (1, 3, 3, 1) 맨 앞에 1이 instence의 개수, 나머지 3 x 3 x 1(채널)

########################################################################

weight = tf.constant([[[[1.]], [[1.]]], [[[1.]], [[1.]]]])
# Filter = W
print(weight.shape)

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')
# input(여기서는 image) : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
# filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.
# strides : 크기 4인 1차원 리스트. [0], [3]은 반드시 1. 일반적으로 [1], [2]는 같은 값 사용.
# padding : 'SAME' 또는 'VALID'. 패딩을 추가하는 공식의 차이.
# SAME은 Zero paddings로서 주위를 0으로 만든다. 출력 크기를 입력과 같게 유지.
# VALID는 패딩 없이 하는 것.

# ex)
# 3x3x1 필터를 32개 만드는 것을 코드로 표현하면 [3, 3, 1, 32]가 된다.
# 순서대로 너비(3), 높이(3), 입력 채널(1), 출력 채널(32)을 뜻한다. 32개의 출력이 만들어진다.

conv2d_img = conv2d.eval()
# eval로 이미지를 실행시킨다.
# 앞에서 InteractiveSession로 실행했기 때문에 가능한 것.
print(conv2d_img.shape)

# 시각화를 위해서 만든 code
conv2d_img = np.swapaxes(conv2d_img, 0 , 3)

for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1, 2, i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    plt.show()

#####################################################################################33

# 필터를 3개로 적용 (2,2,1,3)
weight1 = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]], [[[1., 10., -1.]], [[1., 10., -1.]]]])
print(weight1.shape)

conv2d1 = tf.nn.conv2d(image, weight1, strides=[1,1,1,1], padding='SAME')
conv2d1_img = conv2d1.eval()
print(conv2d1_img.shape)

conv2d1_img = np.swapaxes(conv2d1_img, 0, 3)
for i, one_img in enumerate(conv2d1_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 3, i + 1), plt.imshow(one_img.reshape(3, 3), cmap='gray')
    # subplot(1, 3(여기가 필터의 개수자리, 필터와 동일하게 만들어줘야 한다.))
    plt.show()

# 여기까지 Convolution end
#####################################################################33

# Max Pooling
image1 = np.array([[[[4], [3]], [[2], [1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image1, ksize=[1, 2, 2, 1], strides=[1,1,1,1], padding='SAME')
# ksize는 kernel size = W, filter size를 말한다. 맨 앞에 1이 개수, 2x2x1
print(pool.shape)
print(pool.eval())

#########################################################################

# MNIST image loading
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
img = mnist.train.images[0].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

##########################################################################

# MNIST Convolution layer
sess = tf.InteractiveSession()

img = img.reshape(-1, 28, 28, 1)
# 여기서 -1로 한다면 개수를 하나씩 투입하겠다?라는 느낌으로 집어넣는다.
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))
# 3x3x1는 W의 모양 / 5는 필터의 개수
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='SAME')
# 2x2씩 이동하겠다 0, 3은 건들지 말것.
print(conv2d)

sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)

for i, one_img in enumerate(conv2d_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(14, 14), cmap='gray')
    # plt.show()

#####################################################################################

# MNIST max_pooling
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# conv2d는 1x14x14x5
print(pool)

sess.run(tf.global_variables_initializer())

pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)

for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i + 1), plt.imshow(one_img.reshape(7, 7), cmap='gray')
    plt.show()
