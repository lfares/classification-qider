from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

AUTOTUNE = tf.data.experimental.AUTOTUNE

# import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pathlib

import os

# Auxiliar functions
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')
      plt.show()

class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

PATH = os.getcwd()
data_dir = pathlib.Path(PATH+'/QIDER/train')

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])

print(CLASS_NAMES)

image_count = len(list(data_dir.glob('*/*.jpg')))
# The 1./255 is to convert from uint8 to float32 in range [0,1].
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

image_batch, label_batch = next(train_data_gen)

# Test pre-processed data
# show_batch(image_batch, label_batch)

# Download headless model
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)
feature_extractor_layer.trainable = False


# Attach classification head
model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(train_data_gen.num_classes, activation='softmax')
])

model.summary()

predictions = model(image_batch)

predictions.shape

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])

steps_per_epoch = np.ceil(train_data_gen.samples/train_data_gen.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(train_data_gen, epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'qider_trained_model.h5'
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

