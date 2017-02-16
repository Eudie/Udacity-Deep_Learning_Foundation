#!/usr/bin/python3

import tensorflow as tf
import time

with tf.device('/cpu:0'):
    x1 = tf.Variable(tf.random_normal([10000, 10000]))
    x2 = tf.Variable(tf.random_normal([10000, 10000]))
    result = tf.matmul(x1, x2)
    init = tf.variables_initializer([x1, x2])
    with tf.Session() as sess:
        start_time_1 = time.time()
        sess.run(init)
        output = sess.run([result])
        duration_1 = time.time() - start_time_1
        print("CPU", duration_1)

with tf.device('/gpu:0'):
    x3 = tf.Variable(tf.random_normal([10000, 10000]))
    x4 = tf.Variable(tf.random_normal([10000, 10000]))
    result1 = tf.matmul(x3, x4)
    init1 = tf.variables_initializer([x3, x4])
    with tf.Session() as sess:
        start_time_2 = time.time()
        sess.run(init1)
        output1 = sess.run([result1])
        duration_2 = time.time() - start_time_2
        print("GPU", duration_2)
