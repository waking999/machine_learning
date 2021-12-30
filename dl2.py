import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

path = os.path.dirname(__file__)
print(path)
model = tf.keras.models.load_model(path + '/my_model.h5')

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

predictions = model.predict(test_images)
x_test = test_images[150]
y_test = test_labels[150]
y_pred = np.array(predictions[150])
np.set_printoptions(precision=3, suppress=True)

plt.imshow(x_test, cmap=plt.get_cmap('gray'))
plt.show()
print(y_pred)
