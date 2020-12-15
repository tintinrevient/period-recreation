import tensorflow as tf
import numpy as np
import os
import PIL.Image
import imageio


# Preprocess an image and read it into a NumPy array.
def preprocess(image_path, max_dim=None):
    img = PIL.Image.open(image_path)

    if max_dim:
        img.thumbnail((max_dim, max_dim))

    return np.array(img)


# Normalize an image
def deprocess(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)


def save_img(img, fname):
    imageio.imwrite(fname, img)


# The loss is the sum of the activations in the chosen layers.
# The loss is normalized at each layer so the contribution from larger layers does not outweigh smaller layers.
# Normally, loss is a quantity you wish to minimize via gradient descent.
# In DeepDream, you will maximize this loss via gradient ascent.
def calc_loss(img, model):
    # Pass forward the image through the model to retrieve the activations.
    # Converts the image into a batch of size 1.
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)

    if len(layer_activations) == 1:
        layer_activations = [layer_activations]

    losses = []
    for layer_activation in layer_activations:
        loss = tf.math.reduce_mean(layer_activation)
        losses.append(loss)

    return tf.reduce_sum(losses)


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")

        loss = tf.constant(0.0)

        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


def run_deep_dream_simple(img, deepdream, steps=100, step_size=0.01):

    # Convert from uint8 to the range expected by the model.
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.convert_to_tensor(img)

    step_size = tf.convert_to_tensor(step_size)
    run_steps = tf.constant(steps)

    loss, img = deepdream(img, run_steps, tf.constant(step_size))

    result = deprocess(img)

    return result


if __name__ == "__main__":

    # Base model
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

    # Maximize the activations of these layers
    names = ['mixed2', 'mixed3', 'mixed4', 'mixed5']
    layers = [base_model.get_layer(name).output for name in names]

    # Create the feature extraction model
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

    # Gradient ascent object
    deepdream = DeepDream(dream_model)

    # Original image
    base_image_name = 'green_field.jpg'

    base_image_path = os.path.join('data', base_image_name)
    original_img = preprocess(base_image_path)

    # Gradient ascent steps
    dream_img = run_deep_dream_simple(img=original_img, deepdream=deepdream, steps=50, step_size=0.01)
    # Save the final image
    save_img(dream_img, os.path.join('output', base_image_name))