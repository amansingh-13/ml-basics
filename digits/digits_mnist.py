import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images/255.0
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images/255.0
test_images = test_images.reshape(10000, 28, 28, 1)

model = keras.Sequential([
    keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
	keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=7)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("\nTest accuracy: %s " % test_acc)
# this will help us ensure the saved model is okay for use by the gui

model.save('digits/weights.h5')


