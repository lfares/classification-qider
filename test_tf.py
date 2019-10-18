import tensorflow as tf
import tensorflow_hub as hub
import keras
import numpy as np
import os
import matplotlib.pylab as plt
import pathlib
from PIL import Image

PATH = os.getcwd()
data_dir = pathlib.Path(PATH+'/QIDER/test')

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
class_names = sorted(train_data_gen.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print("Class names:", class_names)

#model = keras.models.load_model("qider_trained_model_20.h5")
#model = tf.keras.experimental.load_from_saved_model('qider_trained_model_20.h5', custom_objects={'KerasLayer':hub.KerasLayer})
model = tf.keras.models.load_model('qider_trained_model_20.h5',custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()
print(model.get_config())

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(label_batch, axis=-1)
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(7):
	plt.subplot(6,5,n+1)
	plt.imshow(image_batch[n])
	color = "green" if predicted_id[n] == label_id[n] else "red"
	plt.title(predicted_label_batch[n].title(), color=color)
	plt.axis('off')
	_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
	plt.show()
