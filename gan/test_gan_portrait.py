import keras
from keras import layers
import numpy as np

# Generator
latent_dim = 32
height = 96
width = 96
channels = 3

generator_input = keras.Input(shape=(latent_dim,))

# First, transform the input into a 48x48 128-channels feature map
x = layers.Dense(128 * 48 * 48)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((48, 48, 128))(x)

# Then, add a convolution layer
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Upsample to 96x96
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)

# Few more conv layers
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)

# Produce a 32x32 1-channel feature map
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

generator = keras.models.Model(generator_input, x)
generator.summary()

# Discriminator
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)

# One dropout layer - important trick!
x = layers.Dropout(0.4)(x)

# Classification layer
x = layers.Dense(1, activation='sigmoid')(x)

discriminator = keras.models.Model(discriminator_input, x)
discriminator.summary()

# To stabilize training, we use learning rate decay
# and gradient clipping (by value) in the optimizer.
discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

# GAN
# Set discriminator weights to non-trainable
# (will only apply to the `gan` model)
discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# Training
import os
from keras.preprocessing import image

dataset_dir_path = "./images"

data_generator = image.ImageDataGenerator(
    rescale=1. / 255,
)

train_generator = data_generator.flow_from_directory(
    directory=dataset_dir_path,
    target_size=(96, 96),
    batch_size=1,
    class_mode="categorical"
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

print("Training data shape:", x_train.shape)

iterations = 550
batch_size = 20
save_dir = './output/'

# Start training loop
import time
start = 0
for step in range(iterations):
    start_time = time.time()
    print("Step ", step);
    # Sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Decode them to fake images
    generated_images = generator.predict(random_latent_vectors)

    # Combine them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    # Assemble labels discriminating real from fake images
    labels = np.concatenate([np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    # Add random noise to the labels - important trick!
    labels += 0.05 * np.random.random(labels.shape)

    # Train the discriminator
    d_loss = discriminator.train_on_batch(combined_images, labels)

    # sample random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    # Assemble labels that say "all real images"
    misleading_targets = np.zeros((batch_size, 1))

    # Train the generator (via the gan model,
    # where the discriminator weights are frozen)
    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    elapsed_time = time.time() - start_time
    print("Duration:", elapsed_time)

    # Occasionally save / plot
    if step % 100 == 0:
        # Save model weights
        gan.save_weights(os.path.join('weights', 'portrait_gan.h5'))

        # Print metrics
        print('discriminator loss at step %s: %s' % (step, d_loss))
        print('adversarial loss at step %s: %s' % (step, a_loss))

        # Save one generated image
        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_portrait' + str(step) + '.png'))

        # Save one real image, for comparison
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_portrait' + str(step) + '.png'))

generator.save(os.path.join('models', 'portrait_generator.h5'))
discriminator.save(os.path.join('models', 'portrait_discriminator.h5'))
gan.save(os.path.join('models', 'portrait_gan.h5'))
