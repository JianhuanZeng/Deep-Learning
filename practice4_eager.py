######################
# the learning lab
# the original code from Columbia University course: COMS 4995 Applied Deep Learning
######################

##############################################################################
############################### What’s a graph? ##############################
#####################
# TensorBoard
import tensorflow as tf
a = tf.constant(2, name=’a’)
b = tf.constant(3, name=’b’)
x = tf.add(a, b)
# $ tensorboard --logdir=graphs/
# then browse to http://localhost:6006
writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))
writer.close()
######################
# Variables must be initialized before use
import tensorflow as tf
w = tf.Variable(10)
with tf.Session() as sess:
    sess.run(w.initializer)
    print(sess.run(w)) # 10
#
w = tf.Variable(10)
with tf.Session() as sess:
    sess.run(w.initializer)
    w.assign(100)
    print(sess.run(w)) # 10
#
w = tf.Variable(10)
with tf.Session() as sess:
    sess.run(w.initializer)
    assign_op = w.assign(100)
    sess.run(assign_op)
    print(sess.run(w)) # 100
#####################
# This code is not deterministic
X, y = tf.Variable(1.0), tf.Variable(1.0)
add_op = x.assign(x + y)
div_op = y.assign(y / 2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for iteration in range(50):
        sess.run([add_op, div_op])
        print(sess.run(w)) # run 1: 2.0, run 2: 2.75
# Feeding and fetching
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)
add_op = tf.add(a,b)
with tf.Session() as sess:
    print(sess.run(add_op, feed_dict={a:[1, 2, 3]}))
#####################
# TensorFlow debugger
# Personal opinion: Skip
#####################
# Graph-based optimizations (XLA)
# A really cool idea, useful for tuning models
# if performance is your bottleneck
# - not necessary otherwise

###############################################################################
################################## Keras ######################################
#####################
# If it feels imperative.
from tensorflow import keras
model = tf.keras.models.Sequential([
tf.keras.layers.Dense(100, activation="relu", input_shape=[10]),
tf.keras.layers.Dense(100),
tf.keras.layers.Dense(1)])
model.weights
[<tf.Variable 'dense_8/kernel:0' shape=(10,100) dtype=float32, numpy=array([...])]
#####################
# Usually
import tensorflow import keras
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001,
patience=0, verbose=1) as tf
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model = tf.keras.models.Sequential([ tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
#####################
# Early stopping
from tensorflow import keras
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
    min_delta=0.001, patience=0, verbose=1) # create a callback
# min_delta: minimum change in the monitored quantity to qualify as an improvement
# patience: number of epochs with no improvement after which training will be stopped.
history = model.fit(x_train, y_train, batch_size=batch_size,
    epochs=epochs, verbose=1, validation_data=(x_test, y_test),
    callbacks=[early_stopping]) # train the model

#####################
# Dropout
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(rate=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(num_classes, activation='softmax'))
# Rate gives the fraction of units that will be randomly dropped at each iteration,
# typically between 0.2 and 0.5.

###############################################################################
################################## eager mode #################################
import tensorflow as tf
#####################
# Enabling
# must be called once at program startup
tf.enable_eager_execution()
# Multiplying a matrix by itself
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(tf.matmul(a, a))
#####################
# NumPy compatibility
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
foo = tf.matmul(a, a)
print(foo.numpy())
# array([[ 7., 10.], [15., 22.]], dtype=float32)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
foo = tf.matmul(a, a)
bar = foo.numpy()
tf.reduce_sum(bar)
# <tf.Tensor: id=58, shape=(), dtype=float32, numpy=54.0>
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(type(a)) # <type 'EagerTensor'>
bar = np.dot(a,a)
print(type(bar)) # <type 'numpy.ndarray'>
#####################
# Gradients
@ops.RegisterGradient("Square")
def _SquareGrad(op, grad):
    x = op.inputs[0]
    y = constant_op.constant(2.0, dtype=x.dtype)
    return math_ops.multiply(grad, math_ops.multiply(x, y))
# Derivative of a function
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
def f(x):
    return tf.square(x)
grads = tfe.gradients_function(f)
grads(3.0) # 6.0
# Higher-order derivatives
def f(x):
    return tf.square(x)
def grad(f):
    return lambda x: tfe.gradients_function(f)(x)[0]
grad(f)(2) # 6.0
grad(grad(f))(2) # 2.0
#####################
# Gradient tapes
c = tfe.Variable([[2.0]])
d = tfe.Variable([[3.0]])
with tf.GradientTape() as tape:
    loss = c*d
grad = tape.gradient(loss, d)
print(grad) # 2.0
# A simple model
class Modeli(object):
    def __init__(self):
        self.W = tfe.Variable(5.0)
        self.b = tfe.Variable(0.0)
    def __call__(self, x):
        return self.W * x + self.b
model = Modeli()
model(3.0).numpy() #15.0
# Loss
def loss(pred, target):
    return tf.reduce_mean(tf.square(pred - target))
loss(2, 3) # 1.0
# It’s eager, it’s Python, it just works.
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000
inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise
print(loss(model(inputs), outputs).numpy())
# Layers are objects (with a call method)
layer = tf.keras.layers.Dense(4, input_shape=(None, 5))
layer(tf.zeros([2, 5]))
layer.variables
[<tf.Variable 'dense_10/kernel:0' shape=(5, 4) dtype=float32, numpy= array(
[[-0.13004386, 0.27818096, 0.7756481 , -0.7271814 ],
[-0.7509823 , -0.09562033, 0.6122887 , -0.7437821 ],
[ 0.16538173, 0.09110129, -0.59920806, 0.49907005],
[ 0.51406395, -0.11622143, -0.14226937, 0.5033473 ],
[-0.5977996 , 0.77903664, 0.6189532 , 0.12730217]], dtype=float32)>,
<tf.Variable 'dense_10/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
#####################
# Extending tf.keras.model
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense_1 = tf.keras.layers.Dense(32)
        self.dense_2 = tf.keras.layers.Dense(10)
    def call(self, inputs):
        x = self.dense_1(inputs)
        x = tf.nn.relu(x)
        return self.dense_2(x)
# a modern training loop
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
for (batch, (images, labels)) in enumerate(dataset):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        loss_value = loss(prediction, labels)
        tf.contrib.summary.scalar('loss', loss_value)
    grads = tape.gradient(loss_value, model.variables)
    optimizer.apply_gradients(zip(grads, model.variables))
#####################
# Switch between eager and graph mode
tf.enable_eager_execution()
print(tf.executing_eagerly()) # True

graph = tf.Graph()
with graph.as_default():
    print(tf.executing_eagerly()) # False
#####################
# Use defun (or define graph-level) functions
import tensorflow as tf
tf.enable_eager_execution()
@tf.contrib.eager.defun
def square_sum(x, y):
    return tf.square(x+y)
result = square_sum(2., 3.)
print(result.numpy()) # 25

#####################
# autograph
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.contrib import autograph

def fib(n):
    a, b, = 0, 1
    for i in range(n-1):
        a,b = b, a+b
    return b
fib(10) #55
graph_fib = autograph.to_graph(fib)
# graph_fib also works in graph mode
result = graph_fib(10)
print(result) # 55
