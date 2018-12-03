############################# the Embedding Projector ################################
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

############################
imdb = keras.datasets.imdb
vocabulary_size = 1000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=vocabulary_size)

# pad the length
max_len = 250
train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=max_len)
test_data = keras.prepocessing.sequence.pad_sequences(test_data, maxlen=max_len)


############################
# build the model
embedding_dimension = 16

model = keras.models.Sequential()
model.add(keras.layers.Embedding(vocabulary_size, embedding_dimension, input_length = max_len))
#model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Dense(1,activation = 'sigmoid'))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile('adam','binary_crossentropy',metrics=['accuracy'])
history = model.fit(train_data, train_labels,
                    epochs=10,
                    batch_size=32,
                    validation_split = 0.2)




############################
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


e = model.layers[0]
weights = e.get_weights()[0]

############################
# visualization
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

############################ shape ############################
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

############################ js ############################
USER_NAME,TOKEN, SITE_NAME =  input()
import os
repo_path = USER_NAME + '.github.io'
if not os.path.exits(os.path.join(os.getcwd(), repo_path)):
    print()
    #! git clone https://{USER_NAME}:{TOKEN}@github.com/{USER_NAME}/{USER_NAME}.github.io
os.chdir(repo_path)

# Create a folder for your site.
project_path = os.path.join(os.getcwd(), SITE_NAME)
if not os.path.exists(project_path):
    os.mkdir(project_path)
os.chdir(project_path)

MODEL_DIR = os.path.join(project_path, "model_js")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

############################
# A few snippets from Alice in Wonderland
ex1 = "Alice was beginning to get very tired of sitting by her sister on the bank."
ex2 = "Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it."

# Dracula
ex3 = "Buda-Pesth seems a wonderful place."
ex4 = "Left Munich at 8:35 P. M., on 1st May, arriving at Vienna early next morning."

# Illiad
ex5 = "Scepticism was as much the result of knowledge, as knowledge is of scepticism."
ex6 = "To be content with what we at present know, is, for the most part, to shut our ears against conviction."

x_train = [ex1, ex2, ex3, ex4, ex5, ex6]
y_train = [0, 0, 1, 1, 2, 2] # Indicating which book each sentence is from



############################
max_len = 20
num_words = 1000

from keras.preprocessing.text import Tokenizer
t = Tokenizer(num_words = 1000)
t.fit_on_texts(x_train)
vectorized = t.texts_to_sequences([ex1])
print(t.word_index)
print(vectorized)
from keras.preprocessing.sequence import pad_sequences
padded = pad_sequences(vectorized, maxl = 20, padding='post')
print(padded)
metadata = {'word_index': t.word_index,
            'max_len': max_len,
            'vocabulary_size': num_words}

############################
embedding_dimension = 8
n_classes = 3
epochs = 10

import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(num_words, embedding_dimension, input_shape=(max_len, )))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile('adam', 'sparse_catergorical_entropy', ['accuracy'])
model.summary()

x_train = t.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post')
model.fit(x_train, y_train, epochs=epochs)

############################
test_example = "Left Munich at 8:35 P. M., on 1st May, arriving at Vienna early next morning."
x_test = t.texts_to_sequences([test_example])
x_test = pad_sequences(x_test, maxlen=max_len, padding='post')
pred = model.predict(x_test)

############################
import json
import tensorflowjs as tfjs

metadata_json_path = os.path.join(MODEL_DIR, 'metadata.json')
json.dump(metadata, open(metadata_json_path, 'wt'))
tfjs.converters.save_keras_model(model, MODEL_DIR)
print('\nSaved model artifcats in directory: %s' % MODEL_DIR)


############################
index_html = """
<!doctype html>
<body>
</body>
"""
with open('index.html','w') as f:
    f.write(index_html)

index_js = """
const HOSTED_URLS ={
    status('Standing by.');
}
setup();
"""
with open('index.js','w') as f:
    f.write(index_js)

############################
import csv
colors_rgb = []
csv_reader = csv.reader(open('color.csv'), delimiter=',')
next(csv_reader)

for row in csv_reader:
    name,r,g,b = row[0].lower().strip(), float(row[1])/255.0, float(row[2])/255.0, float(row[3])/255.0
    colors_rgb.append(( name,r,g,b ))
print(len(colors_rgb), 'colors downloaded')
print('For example', colors_rgb[0])

sentences = []
for row in colors_rgb:
    line = ' '.join([str(part) for part in row])
    sentences.append(line)
