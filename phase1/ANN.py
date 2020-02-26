# Le but de cette exercice est d'apprendre au model d'un ANN comment faire une multiplication par 2 d'un nombre donnée

# build a model that can predict the result of the input x by y that are given

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

# ====================================================================================================
# Settings pour les hyperparametres et les datasets
# ====================================================================================================

EPOCH = 10000
learning_rate = 0.01

# Set une list qui a 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 et Y leurs double
x = [i for i in range(1,10)]
y = [2 * i for i in x]

# ====================================================================================================
# initialize tensorflow placeholders and variables
# ====================================================================================================

# TODO : creer 2 placeholders X et Y qui vont contenir chaque une variable du type float ( float 32 )

X = "VOS CODE ICI"
Y = "VOS CODE ICI"

# TODO : initialiser 2 tensorflow variable d'une maniere random qui seront les weights et les bias du network

W = "VOS CODE ICI"
B = "VOS CODE ICI"

# ====================================================================================================
# initialize tensorflow placeholders and variables
# ====================================================================================================

linear = "ADAPTEZ LA FORMULE DU LINEAR REGRESSION (W * X) + B"
activation = "METTEZ UNE ACTIVATION FONCTION SUR TENSORFLOW"
cost = "RENSEIGNEZ VOUS SUR LES COST FONCTIONS DU NEURAL NETWORK POUR TROUVER LE COST"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# ====================================================================================================
# create a session and run
# ====================================================================================================

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCH):
        batch_id = epoch % 9
        _, loss = sess.run([optimizer, cost], feed_dict={X:x[batch_id], Y:y[batch_id]})
        print(loss)
    print("done")
    print("f(x) = ", sess.run(W)[0] ," * x + ", sess.run(B)[0])

    while True:
        got = input("num:")
        print(sess.run(linear, feed_dict={X:got}))
