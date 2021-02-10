import os
from keras.models import load_model
from keras.preprocessing import image
from keras import models
import numpy as np
import matplotlib.pyplot as plt


# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
num_of_layers = 6 # till 'max_pooling2d_1'
layer_index = 1 # 'separable_conv2d_1'
filter_index = 5 # 3 or 5

# color channel
# color_model = 'rgb'
color_model = 'grayscale'

# path to the dataset
dataset_dir_path = os.path.join('data', 'test')
# portrait
img_path = os.path.join(dataset_dir_path, 'portrait', '0a0c3e63b99b27544f2f440d5d967e2ac.jpg')
# landscape
# img_path = os.path.join(dataset_dir_path, 'landscape', '0a13285ce6b2f9bfe86ac48c278a7878c.jpg')

# path to the trained models
model_filename = os.path.join('model', 'cnn_' + color_model + '_model.h5')
model = load_model(model_filename)
model.summary()


# load the image tensor
img = image.load_img(img_path, target_size=(96, 96), color_mode=color_model)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

# show the input image tensor
if color_model == 'rgb':
    plt.imshow(img_tensor[0])
if color_model == 'grayscale':
    plt.imshow(img_tensor[0], cmap='gray')
plt.show()

# layer outputs
# first layer indices: 0 till 6
layer_outputs = [layer.output for layer in model.layers[:num_of_layers]]

# layer names
for layer in model.layers:
    print(layer.name)

activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# first layer = separable_conv2d
# filter index = 2 --> edge?
first_layer_activation = activations[layer_index]

# layer_index + filter_index
if color_model == 'rgb':
    plt.matshow(first_layer_activation[0, :, :, filter_index], cmap='viridis')
if color_model == 'grayscale':
    plt.matshow(first_layer_activation[0, :, :, filter_index], cmap='gray')
plt.show()