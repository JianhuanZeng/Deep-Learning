"""
# the learning lab
# the original code from Columbia University course: ECBM E4040 Neural Networks and Deep Learning (Prof.Zoran Kostić) 
"""
import tensorflow as tf
import numpy as np
mnist = input_data.read_data_sets('./tmp/data', one_hot=True)
num_train = 55000


batch_size = 50
epochs = 50
lr = 0.01 # Learning rate
Xte = mnist.test.images # Test data
Xte = mnist.test.lables


# Explicitly set variables in the gpu memory.
# If you don't have a GPU,
# comment the 'with tf.device('/gpu"0')' line,
# and remove the following indents.
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32,[None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')
    W = tf.Variable(tf.random_uniform([784,10]),dtype=tf.float32, name='weights') # Model weights (trainable)
    b = tf.Variable(tf.random_uniform(1,10), dtype=tf.float32, name='bias') # Model bias (trainable)
    pred = tf.matmul(x,W)+b
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred), name='loss') # Model loss function
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss) # Use basic gradient descent optimizer
        test_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) # Test whether the predictions match real labels
        accuracy = tf.reduce_mean(tf.cast(test_prediction, tf.float32))
        )


# Using minibatch to train the model.
num_batch = num_train/batch_size
init = tf.global_variable_initializer()

# Train the model
with tf.Session as sess:
    sess.run(init)
    for epoch in range(epochs):
        cost_this_epoch = 0
        for i in range(int(num_batch)):
            xtr,ytr = mnist.train.next_batch(batch_size)
            _,l = sess.run([optimizer, loss],feed_dict={x:xtr, y:ytr})
            cost_this_epoch += 1*batch_size
        print('Epoch {} done. Loss:{:5f}'.format(epoch,cost_this_epoch))
    accr = sess.run(accuracy, feed_dict={x:Xte, y:Yte})
print('accuracy is {}%'.format(accr*100))
