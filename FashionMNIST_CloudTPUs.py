######################
# the learning lab
# original code from https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/fashion_mnist.ipynb
######################

import tensorflow as tf
import numpy as np

##################################### Defining the model ###########################################
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test, -1 )

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(64,(5,5),padding='same',activation='elu'))
model.add(tf.keras.layers.MaxPooling2D(pooling_size=(2,2),strides=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(128,(5,5),padding='same',activation='elu'))
model.add(tf.keras.layers.MaxPooling2D(pooling_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(256,(5,5),padding='same',activation='elu'))
model.add(tf.keras.layers.MaxPooling2D(pooling_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Activation('elu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Activation('softmax'))

model.summary()
################################# Training on the TPU ###############################################
import os
tpu_model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://'+os.environ['COLAB_TPU_ADDR'])))
tpu_model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-3), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['sparse_categorical_accuracy'])
def train_gen(batch_size):
    while True:
        offset = np.random.randint(0, x_train.shape[0]-batch_size)
        yield x_train[offset:offset + batch_size], y_train[offset:offset + batch_size]
# yield is handy when your function will return a huge set of values that you will only need to read once.
tpu_model.fit_generator(train_gen(1024), epochs=10, steps_per_epoch=100, validation_data=(x_test, y_test))


################################# Training on the TPU ###############################################

LABEL_NAMES = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']
cpu_model = tpu_model.sync_to_cpu()

from matplotlib import pyplot
%matplotlib inline
def plot_pre(images, predictions):
    n = images.shape[0]
    nc = int(np.ceil(n/4))
    f, axes = pyplot.subplots(nc,4)
    for i in range(nc*4):
        y = i//4
        x = i%4
        axes[x,y].axes('off')

        label = LABEL_NAMES[np.argmax(predictions(i))]
        confidence = np.max(predictions(i))
        if i > n:
            continue
        axes[x,y].imshow(images[i])
        axes[x,y].text(0.5,0.5,label+'\n%.3f' % confidence, fontsize=14)

        pyplot.gcf().set_size_inches(8,8)


plot_pre(np.squeeze(x_test[:16]), cpu_model.predict(x_test[:16]))
