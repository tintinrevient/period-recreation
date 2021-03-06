import os
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow import keras


# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_name = "separable_conv2d_1"
filter_index = 1

# color channel
# rgb
# color_channel = 3
# color_model = 'rgb'

# grayscale
color_channel = 1
color_model = 'grayscale'

# The dimensions of our input image
img_width = 96
img_height = 96

# path to the trained models
model_filename = os.path.join('model', 'cnn_' + color_model + '_model.h5')
model = load_model(model_filename)
model.summary()

# Set up a model that returns the activation values for our target layer
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)


def compute_loss(input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img


def initialize_image():
    # We start from a gray image with some random noise
    # color image
    img = tf.random.uniform((1, img_width, img_height, color_channel))
    # gray image
    # img = tf.random.uniform((1, img_width, img_height))

    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25


def visualize_filter(filter_index):
    # We run gradient ascent for 20 steps
    iterations = 30
    learning_rate = 10.0
    img = initialize_image()
    for iteration in range(iterations):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate)

    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


# first layer = separable_conv2d
# filter index = 2 --> edge?
loss, img = visualize_filter(filter_index)
keras.preprocessing.image.save_img("pix/" + layer_name + "_filter_" + str(filter_index) + "_" + color_model + ".png", img)