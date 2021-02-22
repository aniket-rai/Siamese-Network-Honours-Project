#!/usr/bin/env python

# Following along with: https://www.tensorflow.org/tutorials/quickstart/beginner

import tensorflow as tf

# Load and prepare MNIST. Convert from integers to floating point numbers
# (normalize)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# build the model by stacking layers. choose an optimizer and loss function

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# Show what the model looks like before training
#
# While we cound add the softmax directly to the end of the model, this isn't
# great, as it can't provide exact or numerically stable loss calculations
predictions = model(x_train[:1]).numpy()
print('Initial probabilities for each class: ', tf.nn.softmax(predictions).numpy())

# Create the loss function from keras' built-in ones
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# We expect this to be around `-log(1.10) ~= 2.3`
print('Initial loss: ', loss_fn(y_train[:1], predictions).numpy())

# Set up our optimizer, loss function,ready to fit
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Fit model, adjusting the parameters to minimize loss
model.fit(x_train, y_train, epochs=5)

# check the model's performance, typically on a validation/test set
print(model.evaluate(x_test,  y_test, verbose=2))

# If everything's gone well, the model is now trained to 98% accuracy on this
# dataset.

# If you want a model that returns a probability, you can wrap the trained
# model, and then attach the softmax
prob_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
