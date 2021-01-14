import numpy as np
from scipy.stats import norm
from keras import models
import os

decoder = models.load_model(os.path.join('models', 'decoder_portraits.h5'))

# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 96
figure = np.zeros((digit_size * n, digit_size * n, 3))
batch_size = 16

# Linearly spaced coordinates on the unit square were transformed
# through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z,
# since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

def deprocess_image(img):

    # Normalize array: center on 0., ensure variance is 0.15
    # img -= img.mean()
    # img /= img.std() + 1e-5

    # Clip to [0, 1]
    # img += 0.5
    # img = np.clip(img, 0, 1)

    # Convert to 255 array
    img *= 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size=batch_size)

        digit = x_decoded[0].reshape(digit_size, digit_size, 3)
        digit = deprocess_image(digit)

        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit


import imageio
imageio.imwrite(os.path.join('output', 'latent_space_portraits.png'), figure.astype(np.uint8))