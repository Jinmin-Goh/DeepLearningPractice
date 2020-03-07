# Lab 8
# Made by:  Jinmin Goh
# Date:     20200304

# Tensor Manipulation

# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# 1D Array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

# 2D Array
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape

# Shape, Rank, Axis
t = tf.constant([1,2,3,4])  # rank: 1
tf.shape(t).eval()

t = tf.constant([[1,2],
                 [3,4]])    # rank: 2
tf.shape(t).eval()

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
    # rank: 4
tf.shape(t).eval()
tf.shape
    # the outmost axis is 0, inmost axis is 3 (= -1) which has raw numbers

# Matmul vs. multiply
matrix1 = tf.constant([[3., 4.]])
matrix2 = tf.constant([[2.],[1.]])
print("Matrix 1 shape", matrix1.shape)
print("Matrix 2 shape", matrix2.shape)
tf.matmul(matrix1, matrix2).eval()

(matrix1*matrix2).eval()    # wrong multiplying for matrix

# Broadcasting
# Broadcasting makes adding or multiplying for differnt shape matrices
    # adds same shape matrix
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1+matrix2).eval()
    # adds differnt shape matrix
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
(matrix1+matrix2).eval()

# Random values for variable initializations
tf.random_normal([3]).eval()        # normal distribution; mean = 0, SD = 1
tf.random_uniform([2]).eval()       # uniform distribution
tf.random_uniform([2, 3]).eval()

# Reduce mean/sum
tf.reduce_mean([1, 2], axis = 0).eval() # 1, because integer
x = [[1., 2.],      #  ________ axis 1
     [3., 4.]]      # |
                    # |
                    # axis 0
tf.reduce_mean(x).eval()            # 2.5 (total mean value)
tf.reduce_mean(x, axis = 0).eval()  # array([2., 3.], dtype=float32)
tf.reduce_mean(x, axis = 1).eval()  # array([1.5, 3.5], dtype=float32)

tf.reduce_sum(x).eval()             # 10.0 (total sum value)
tf.reduce_sum(x, axis = 0).eval()   # array([4., 6.], dtype=float32)
tf.reduce_sum(x, axis = -1).eval()  # array([3., 7.], dtype=float32)
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()    # 5.0

# Argmax with axis
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis = 0).eval()   # array([1, 0, 0], dtype=int64)
tf.argmax(x, axis = 1).eval()   # array([2, 0], dtype=int64)
tf.argmax(x, axis = -1).eval()  # array([2, 0], dtype=int64)

# Reshape, squeeze, expand_dims
t = np.array([[[0, 1, 2], 
               [3, 4, 5]],
              
              [[6, 7, 8], 
               [9, 10, 11]]])
t.shape     # (2, 2, 3)
tf.reshape(t, shape = [-1, 3]).eval()
"""
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
"""
tf.reshape(t, shape = [-1, 1, 3]).eval()
"""
array([[[ 0,  1,  2]],

       [[ 3,  4,  5]],

       [[ 6,  7,  8]],

       [[ 9, 10, 11]]])
"""
tf.squeeze([[0], [1], [2]]).eval()      # array([0, 1, 2])
tf.expand_dims([0, 1, 2], 1).eval()     # array([[0],[1],[2]])

# One hot
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()
"""
array([[[ 1.,  0.,  0.]],

       [[ 0.,  1.,  0.]],

       [[ 0.,  0.,  1.]],

       [[ 1.,  0.,  0.]]], dtype=float32)
"""
    # one hot automatically expand one rank
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval() # reduced one rank with reshaping
"""
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]], dtype=float32)
"""

# Casting
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()

# Stack
x = [1, 4]
y = [2, 5]
z = [3, 6]
    # Pack along first dim.
tf.stack([x, y, z]).eval()          # (3, 2) array
"""
array([[1, 4],
       [2, 5],
       [3, 6]])
"""
tf.stack([x, y, z], axis=1).eval()  # (2, 3) array
"""
array([[1, 2, 3],
       [4, 5, 6]])
"""

# Ones and Zeros like
    # fills same shape with 0 or 1 -> initializing
x = [[0, 1, 2],
     [2, 1, 0]]
tf.ones_like(x).eval()
tf.zeros_like(x).eval()

# Zip
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
"""
1 4
2 5
3 6
"""
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
"""
1 4 7
2 5 8
3 6 9
"""

# Transpose
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)

t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1))

t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2))

t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t))