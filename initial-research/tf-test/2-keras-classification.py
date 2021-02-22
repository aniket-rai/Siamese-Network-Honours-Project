#!/usr/bin/env python

# Following along with https://www.tensorflow.org/tutorials/keras/classification

import tensorflow as tf

# Helper libs
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# In fashion mnist, images are greyscale 28*28, just like mnist. however, the
# output is a num from 0 - 9: what 'class' of clothing it is.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Data preprocessing

# Turn integer values between 0 and 255 to floats between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show what the data looks like
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# Build the model
#
# Most deep learning is a sequential chain of layers
#
# We're building a very simple, fully connected (aka Dense) network: going from
# 284 neurons to 128 to 10 - so one hidden layer. We also have an input layer
# which transforms the input from 28 * 28 to 784. The output is a 'logits' array
# of length 10, which, indicates that the current image belongs to one of the 10
# classes. If the probability of something, p, is between 0 and 1, the logit
# representation of that is `log(p/(1-p))`

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
#
# - The loss function represents how accurate the model is during training
# - The optimiser determines how the model is updated based on the data and loss
#   function
# - metrics are what are used to monitor training and testing steps.

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
#
# 1. feed training data to the model (train_images and train_labels)
# 2. the model learns to associate images and labels
# 3. ask the model to make predictions about the test set (test_images)
# 4. verify that the predictions match the labels from test_labels

model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy', test_acc)

# If things go to plan, you'll see 91% accuracy on the training data, but less
# on the test data. We're overfitting here. The site includes some links on how
# to prevent it.

# Make predictions

prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = prob_model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100*np.max(predictions_array),
            class_names[true_label]),
        color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# i = 0
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions[i], test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions[i], test_labels)
# plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], test_labels)
# plt.tight_layout()
# plt.show()

# Test for single image
#
# keras works in batches, so you need to turn one image into a batch of size 1
img = test_images[1]
img = (np.expand_dims(img, 0))
predictions_single=prob_model.predict(img)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
print(np.argmax(predictions_single[0]))
