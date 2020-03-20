# Lab 12-0
# Made by:  Jinmin Goh
# Date:     20200321

# RNN basics

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# One cell RNN input_dim (4) -> output_dim (2)
# input shape: (1,1,4) -> [[[1,0,0,0]]] / output shape: (1,1,2) -> [[[x,x]]]
with tf.variable_scope('one_cell') as scope:
    hidden_size = 2     # output size
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
    # cell2 = tf.keras.layers.LSTMCell(units=hidden_size)   # other cell types
    print(cell.output_size, cell.state_size)
    x_data = np.array([[h]], dtype=np.float32) # x_data = [[[1,0,0,0]]]
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

# One cell RNN input_dim (4) -> output_dim (2). sequence: 5
# unfolding to n sequences
# input shape: (1,5,4) / output shape: (1,5,2)
with tf.variable_scope('two_sequances') as scope:
    hidden_size = 2
    cell = tf.keras.layers.SimpleRNNCell(units=hidden_size)
    x_data = np.array([[h, e, l, l, o]], dtype=np.float32)
    print(x_data.shape)
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

# Batching input
# One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
with tf.variable_scope('3_batches') as scope:
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)
    hidden_size = 2
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
    
# One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch 3
# 3 batches 'hello', 'eolll', 'lleel'
with tf.variable_scope('3_batches_dynamic_length') as scope:
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)
    hidden_size = 2
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, x_data, sequence_length=[5,3,4], dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

# Create input data
batch_size=3
sequence_length=5
input_dim=3

x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
pp.pprint(x_data)  # batch, sequence_length, input_dim

# One cell RNN input_dim (3) -> output_dim (5). sequence: 5, batch: 3
with tf.variable_scope('generated_data') as scope:
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data,
                                         initial_state=initial_state, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

with tf.variable_scope('MultiRNNCell') as scope:
    # Make rnn
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 3 layers
    # rnn in/out
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

with tf.variable_scope('dynamic_rnn') as scope:
    cell = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32,
                                         sequence_length=[1, 3, 2])
    # lentgh 1 for batch 1, lentgh 2 for batch 2
    print("dynamic rnn: ", outputs)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())  # batch size, unrolling (time), hidden_size

with tf.variable_scope('bi-directional') as scope:
    # bi-directional rnn
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=5, state_is_tuple=True)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_data,
                                                      sequence_length=[2, 3, 1],
                                                      dtype=tf.float32)
    sess.run(tf.global_variables_initializer())

# flattern based softmax
hidden_size=3
sequence_length=5
batch_size=3
num_classes=5

pp.pprint(x_data) # hidden_size=3, sequence_length=4, batch_size=2
x_data = x_data.reshape(-1, hidden_size)
pp.pprint(x_data)

softmax_w = np.arange(15, dtype=np.float32).reshape(hidden_size, num_classes)
outputs = np.matmul(x_data, softmax_w)
outputs = outputs.reshape(-1, sequence_length, num_classes) # batch, seq, class
pp.pprint(outputs)

# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction = tf.constant([[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=prediction, targets=y_data, weights=weights)
sess.run(tf.global_variables_initializer())
print("Loss: ", sequence_loss.eval())

# [batch_size, sequence_length]
y_data = tf.constant([[1, 1, 1]])

# [batch_size, sequence_length, emb_dim ]
prediction1 = tf.constant([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]], dtype=tf.float32)
prediction2 = tf.constant([[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], dtype=tf.float32)

prediction3 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype=tf.float32)
prediction4 = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32)

# [batch_size * sequence_length]
weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
sequence_loss2 = tf.contrib.seq2seq.sequence_loss(prediction2, y_data, weights)
sequence_loss3 = tf.contrib.seq2seq.sequence_loss(prediction3, y_data, weights)
sequence_loss4 = tf.contrib.seq2seq.sequence_loss(prediction3, y_data, weights)

sess.run(tf.global_variables_initializer())
print("Loss1: ", sequence_loss1.eval(),
      "Loss2: ", sequence_loss2.eval(),
      "Loss3: ", sequence_loss3.eval(),
      "Loss4: ", sequence_loss4.eval())