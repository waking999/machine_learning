import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# x = train_images[0]
# y = train_labels[0]

# plt.imshow(x, cmap=plt.get_cmap('gray'))
# plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

model.save('saved_model/my_model')
model.save('./my_model.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:' + str(test_acc))
print('test_loss:' + str(test_loss))
