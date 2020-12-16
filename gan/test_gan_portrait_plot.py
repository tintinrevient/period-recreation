import matplotlib.pyplot as plt
import numpy as np
from keras import models
import os
from keras.preprocessing import image

latent_dim = 32
generator = models.load_model(os.path.join('models', 'portrait_generator.h5'))

# Sample random points in the latent space
random_latent_vectors = np.random.normal(size=(10, latent_dim))

# Decode them to fake images
generated_images = generator.predict(random_latent_vectors)

def deprocess_image(img):
    # Convert to 255 array
    img *= 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

nrow = 2
ncol = 5
digit_size = 96
margin = 5
figure = np.zeros((digit_size * nrow + (nrow - 1) * margin,
                   digit_size * ncol + (ncol - 1) * margin,
                   3))

for i in range(generated_images.shape[0]):
    # img = image.array_to_img(generated_images[i] * 255., scale=False)
    img = deprocess_image(generated_images[i])

    row = i // 5
    col = i % 5
    figure[row * digit_size + row * margin: (row + 1) * digit_size + row * margin,
           col * digit_size + col * margin: (col + 1) * digit_size + col * margin] = img

figure = image.array_to_img(figure, scale=False)

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
