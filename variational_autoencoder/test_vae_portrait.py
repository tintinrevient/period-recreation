import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

img_shape = (96, 96, 3)
batch_size = 16
latent_dim = 2  # Dimensionality of the latent space: a plane

input_img = keras.Input(shape=img_shape)

x = layers.Conv2D(32, 3,
                  padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(128, 3,
                  padding='same', activation='relu')(x)
x = layers.Conv2D(256, 3,
                  padding='same', activation='relu')(x)

shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])


# This is the input where we will feed `z`.
decoder_input = layers.Input(K.int_shape(z)[1:])

# Upsample to the correct number of units
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# Reshape into an image of the same shape as before our last `Flatten` layer
x = layers.Reshape(shape_before_flattening[1:])(x)

# We then apply then reverse operation to the initial
# stack of convolution layers: a `Conv2DTranspose` layers
# with corresponding parameters.
x = layers.Conv2DTranspose(32, 3,
                           padding='same', activation='relu',
                           strides=(2, 2))(x)
x = layers.Conv2D(3, 3,
                  padding='same', activation='sigmoid')(x)
# We end up with a feature map of the same size as the original input.

# This is our decoder model.
decoder = Model(decoder_input, x)

# We then apply it to `z` to recover the decoded `z`.
z_decoded = decoder(z)


class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

# We call our custom layer on the input and the decoded output,
# to obtain the final model output.
y = CustomVariationalLayer()([input_img, z_decoded])

from keras.datasets import mnist

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

from keras_preprocessing import image

dataset_dir_path = "./images"

data_generator = image.ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = data_generator.flow_from_directory(
    directory=dataset_dir_path,
    target_size=(96, 96),
    batch_size=1,
    class_mode="categorical",
    subset="training"
)

validation_generator = data_generator.flow_from_directory(
    directory=dataset_dir_path,
    target_size=(96, 96),
    batch_size=1,
    class_mode="categorical",
    subset="validation"
)

print("Loading training and validation dataset...")

x_train = []
n = 0
for x, y in train_generator:
    if n < train_generator.n:
        print("Loading training data", n)
        x_train.append(x[0])
        n = n + 1
    else:
        break

x_train = np.asarray(x_train)


x_test = []
n = 0
for x, y in validation_generator:
    if n < validation_generator.n:
        print("Loading validation data", n)
        x_test.append(x[0])
        n = n + 1
    else:
        break

x_test = np.asarray(x_test)

print("Training shape:", x_train.shape)
print("Validation shape:", x_test.shape)

vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=3,
        batch_size=batch_size,
        validation_data=(x_test, None))

# save the model
import os
decoder.save(os.path.join('models', 'decoder_portraits.h5'))


import matplotlib.pyplot as plt
from scipy.stats import norm

# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 96
figure = np.zeros((digit_size * n, digit_size * n, 3))

# Linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z,
# since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

def deprocess_image(img):

    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to 255 array
    img *= 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)

        print(x_decoded[0])

        digit = x_decoded[0].reshape(digit_size, digit_size, 3)
        digit = deprocess_image(digit)

        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit


import imageio
imageio.imwrite(os.path.join('output', 'latent_space_portraits.png'), figure.astype(np.uint8))