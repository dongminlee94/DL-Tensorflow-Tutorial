import tensorflow as tf

# x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
# # x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
# # x = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12], [13, 14, 15, 16]]]])
# # x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
# y = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
#
# a = np.argmax(x, axis=-1)
# b = np.argmax(x, axis=0)
# c = np.argmax(x, axis=1)
# d = np.argmax(x, axis=2)
# e = np.argmax(x, axis=3)
#
# f = np.reshape(y, newshape=[-1, 3])
# g = np.reshape(y, newshape=[-1, 1, 3])
# # g = np.reshape(y, newshape=[-1, 3])
# # g = np.reshape(y, newshape=[-1, 3])
#
# print(f)
# print(g)
#
# # print(a)
# # print("-----------------------------------------")
# # print(b)
# # print("-----------------------------------------")
# # print(c)
# # print("-----------------------------------------")
# # print(d)
# # print("-----------------------------------------")
# # print(e)

####################################################################3

# def cross_entropy_error():
#     y = np.array([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]])
#     b = y.shape[0]
#     print(b)
#     t = np.array([2, 4])
#     print(y[np.arange(b), t])
#
# print(cross_entropy_error())
#
# a = 5
# t = 2
# b = np.arange(a, t)
# print(b)

#########################################################################3
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

# print(x_train.shape)
# print(t_train.shape)
# print(x_test.shape)
# print(t_test.shape)
#
# train_size = x_train.shape[0]
# batch_size = 100
# batch_mask = np.random.choice(train_size, batch_size)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y)) / batch_size






