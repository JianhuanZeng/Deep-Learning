#############################################################################
# build the sensement analysis for movie reviews, from raw text to word embeddings
############################# the Embedding Projector #######################
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

############################
imdb = keras.datasets.imdb
vocabulary_size = 1000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocabulary_size)

# pad the length
max_len = 25
train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_len)
test_data = keras.prepocessing.sequence.pad_sequences(test_data, maxlen=max_len)


############################
# build the sensement analysis model
embedding_dimension = 16

model = keras.models.Sequential()
model.add(keras.layers.Embedding(vocabulary_size, embedding_dimension, input_length = max_len))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile('adam','binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=32,
                    validation_split = 0.2)


############################
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

########################################################
# download 
import os 
import tarfile
from six.moves.urllib import request

url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
filename = './aclImdb_v1.tar.gz'

if not os.path.isfile(filename):
  #!curl -O 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
  request.urlretrieve(url=url, filename=filename)

tar = tarfile.open('aclImdb_v1.tar.gz')
tar.extractall()
tar.close() 

########################################################
# from raw text to word embeddings
word_index = imdb.get_word_index()
word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
reversed_word_index = dict([(i,w) for w,i in word_index.items()])

def decode_word(sent):
    return ''.join([reversed_word_index[i] for i in sent])
decode_word(train_data[3])


############################
e = model.layers[0]
weights = e.get_weights()[0]

############################
# visualization embeddings
out_v = open('vecs.tsc', 'w')
out_m = open('meta.tsv', 'w')
for i in range(vocabulary_size):
    word = reversed_word_index[i]
    embeddings = weights[i]
    out_m.write(word+'\n')
    out_v.write('\t'.join([str(x) for x in embeddings])+'\n')
out_v.close()

from google.colab import files
files.download('vecs.tsc')
files.download('meta.tsc')











#############################################################################
############################ understanding shape ############################
import tensorflow as tf
from tensorflow import keras
tf.enable_eager_execution()

vocabulary_size = 7
embedding_dimension = 8
layer = keras.layers.Embedding(vocabulary_size, embedding_dimension)
tensor = tf.constant([[1,2,3],[4,5,6]])
# (2,3)
result = layer(tensor)
# (2,3,8)
gp = keras.layers.GlobalAveragePooling1D()
# (2,8)
