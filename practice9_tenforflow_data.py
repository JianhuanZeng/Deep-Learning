import tensorflow as tf
tf.enable_eager_execution()
tf.VERSION

import pathlib
data_root = tf.keras.utils.get_file('flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data_root = pathlib.Path(data_root)

for item in data_root.iterdir():
    print(item)
##############################
import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = list(map(lambda x: str(x), all_image_paths))
random.shuffle(all_image_paths)
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
flow_index = dict((name,index) for index,name in enumerate(label_names))
all_image_labels = [flow_index[pathlib.Path(path).parent.name] for path in all_image_paths]
##############################
attributions = (data_root/"LICENSE.txt").read_text().splitlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

import IPython.display as display
def caption_image(image_path):
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    return  "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])
for n in range(3):
    image_path = random.choice(all_image_paths)
    display.display(display.Image(image_path))
    print(caption_image(image_path))
    print()



##############################
img_path = all_image_paths[0]
def preprocess_image(path):
    image = tf.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

import matplotlib.pyplot as plt
image_path = all_image_paths[0]
label = all_image_labels[0]
plt.imshow(preprocess_image(image_path))
plt.grid(False)
plt.xlabel(caption_image(image_path))
plt.title(label_names[label].title())

##############################
# Way 1
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(preprocess_image, num_parallel_calls=8)

plt.figure(figsize=(8,8))
for n,image in enumerate(image_ds.take(4)):
    plt.subplot(2,2,n+1)
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')
    plt.xlabel(caption_image(all_image_paths[n]))

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
for label in label_ds.take(10):
    print(label_names[label.numpy()])


image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
# The simple pipeline used above reads each file individually, on each epoch. This is fine for local training on CPU but may not be sufficient for GPU training, and is totally inapprpriate for any sort of distributed training.
############################## training ##############################
BATCH_SIZE = 32
ds = image_label_ds.repeat()
ds = ds.shuffle(buffer_size=4000)
ds = ds.batch(BATCH_SIZE).prefetch(1)
##############################
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192,192), include_top=False)
mobile_net.trainable = False

# the [-1,1] range
def change_range(image,label):
  image = 2*image-1
  return image, label

keras_ds = ds.map(change_range)

image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)


model = tf.keras.Sequential(mobile_net,
                            tf.keras.layers.GlobalAveragePooling2D(),
                            tf.keras.layers.Dense(len(label_names)))
logit_batch = model(image_batch).numpy()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
model.fit(ds, epochs=1, steps_per_epoch=3)
############################## Performance ##############################
import time
def timeit(ds,batches=100):
    overall_start = time.time()
    it = iter(ds.take(batches + 1))
    next = it
    start = time.time()
    for i, (images, labels) in enumerate(it):
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()
    duration = end-start
    print("{} batches: {} s".format(batches, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
    print("Total time: {}s".format(end-overall_start))
timeit(ds)

############################## Cache ##############################
# Way 2.1
ds = image_label_ds.cache().repeat()
ds = ds.shuffle(buffer_size=4000)
ds = ds.batch(BATCH_SIZE).prefetch(1)
timeit(ds)

# Way 2.2
ds = image_label_ds.cache(filename='./cache.tf-data').repeat()
ds = ds.shuffle(buffer_size=4000)
ds = ds.batch(BATCH_SIZE).prefetch(1)
timeit(ds)

############################## TFRecord ##############################
# Way 3
image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.read_file)
tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
tfrec.write(image_ds)

image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image, num_parallel_calls=8)
ds = tf.data.Dataset.zip((image_ds, label_ds))
ds = ds.repeat().shuffle(buffer_size=4000)
ds=ds.batch(BATCH_SIZE).prefetch(1)
timeit(ds)

