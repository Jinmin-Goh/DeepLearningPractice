# Lab 9-MNIST
# Made by:  Jinmin Goh
# Date:     20200306

# MNIST dataset wide deep NN model with tensorboard
# Acc: 94.2%

import tensorflow as tf
import matplotlib.pyplot as plt
import random
tf.set_random_seed(777)  # for reproducibility
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
#MNIST_size = 784
#hidden_size = 20

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# layer 1: weight number same with MNIST_size
with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random_normal([784, 128]), name='weight1')
    b1 = tf.Variable(tf.random_normal([128]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    # histogram
    tf.summary.histogram("W1", W1)
    tf.summary.histogram("b1", b1)
    tf.summary.histogram("Layer1", layer1)
# layer 2: hidden layer
with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([128, 32]), name='weight2')
    b2 = tf.Variable(tf.random_normal([32]), name='bias2')
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
    tf.summary.histogram("W2", W2)
    tf.summary.histogram("b2", b2)
    tf.summary.histogram("Layer2", layer2)
# layer 3: hidden layer
with tf.name_scope("Layer3"):
    W3 = tf.Variable(tf.random_normal([32, 16]), name='weight3')
    b3 = tf.Variable(tf.random_normal([16]), name='bias3')
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
    tf.summary.histogram("W3", W3)
    tf.summary.histogram("b3", b3)
    tf.summary.histogram("Layer3", layer3)
# layer 4: 10 classifications
with tf.name_scope("Layer4"):
    W4 = tf.Variable(tf.random_normal([16, nb_classes]), name='weight4')
    b4 = tf.Variable(tf.random_normal([nb_classes]), name='bias4')
    hypothesis = tf.nn.softmax(tf.matmul(layer3, W4) + b4)
    tf.summary.histogram("W3", W4)
    tf.summary.histogram("b3", b4)
    tf.summary.histogram("Hypothesis", hypothesis)

# Cross entropy
with tf.name_scope("Cost"):
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
    tf.summary.scalar("Cost", cost)
with tf.name_scope("Train"):
    train = tf.train.GradientDescentOptimizer(learning_rate = 1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.summary.scalar("Accuracy", accuracy)

# parameters
num_epochs = 15     # training count of entinre training data
batch_size = 100    # splitting size of whole dataset
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/MNIST")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0
        for i in range(num_iterations):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, summary, cost_val = sess.run([train, merged_summary, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations
            writer.add_summary(summary, global_step = i + epoch * num_iterations)
        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))
    print("Learning finished")
    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )
    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )
    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()

