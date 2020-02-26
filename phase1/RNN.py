import tensorflow as tf
import numpy as np

# ==============================================================================================================
# hyperparameters
# ==============================================================================================================

lr = 0.005
epoch = 30
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# ==============================================================================================================
# Variables
# ==============================================================================================================

X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([256, 10]))
b = tf.Variable(tf.random_normal([10]))

cell = tf.nn.rnn_cell.BasicRNNCell(256)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
outputs = tf.transpose(outputs, [1, 0, 2])[-1]
final = tf.matmul(outputs, W) + b

print(Y.shape)
print(final.shape)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=final, labels=Y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(final, 1)), tf.float32))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for times in range(epoch):
        xs, ys = mnist.train.next_batch(125)
        xs = xs.reshape((125, 28, 28))
        _, loss , acc= sess.run([optimizer, cost, accuracy], feed_dict={X:xs, Y:ys})
        print(loss, ":", acc)