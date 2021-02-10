import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

"""
## File path
"""

# dataset path
dataset_dir_path = os.path.join('images')
# encoder path
encoder_path = os.path.join('models', 'encoder.h5')

"""
## Prepare the data
"""

# number of classes
num_classes = 50
# target size
target_size = [96, 96]
# input shape
input_shape = (96, 96, 3)

# hyper-parameters
learning_rate = 0.001
batch_size = 265
hidden_units = 512
projection_units = 128
num_epochs = 50
dropout_rate = 0.5
temperature = 0.05


train_data_generator = image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=[0.9, 1.4],
        brightness_range=[0.75, 1.25],
        rotation_range=10
)

training_generator = train_data_generator.flow_from_directory(
        directory=dataset_dir_path,
        target_size=tuple(target_size),
        batch_size=batch_size,
        class_mode="categorical"
)

validation_generator = train_data_generator.flow_from_directory(
        directory=dataset_dir_path,
        target_size=tuple(target_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
)


"""
## Plot the history
"""

def plot_history(history):
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training and validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()


"""
## Build the encoder model
The encoder model takes the image as input and turns it into a 2048-dimensional
feature vector.
"""

def create_encoder():
    resnet = keras.applications.ResNet50V2(
        include_top=False, weights=None, input_shape=input_shape, pooling="avg"
    )

    inputs = keras.Input(shape=input_shape)
    outputs = resnet(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="painter-encoder")
    return model


encoder = create_encoder()
encoder.summary()


"""
## Build the classification model
The classification model adds a fully-connected layer on top of the encoder,
plus a softmax layer with the target classes.
"""

def create_classifier(encoder, trainable=True):

    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(num_classes, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


"""
## Experiment 1: Train the baseline classification model
In this experiment, a baseline classifier is trained as usual, i.e., the
encoder and the classifier parts are trained together as a single model
to minimize the crossentropy loss.
"""

encoder = create_encoder()
classifier = create_classifier(encoder)
classifier.summary()

history = classifier.fit_generator(
        generator=training_generator,
        steps_per_epoch=training_generator.n / batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n / batch_size,
        verbose=1
)

plot_history(history)


"""
## Experiment 2: Use supervised contrastive learning
In this experiment, the model is trained in two phases. In the first phase,
the encoder is pretrained to optimize the supervised contrastive loss,
described in [Prannay Khosla et al.](https://arxiv.org/abs/2004.11362).
In the second phase, the classifier is trained using the trained encoder with
its weights freezed; only the weights of fully-connected layers with the
softmax are optimized.

### 1. Supervised contrastive learning loss function
"""

class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model


"""
### 2. Pretrain the encoder
"""

encoder = create_encoder()

encoder_with_projection_head = add_projection_head(encoder)
encoder_with_projection_head.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=SupervisedContrastiveLoss(temperature),
)

encoder_with_projection_head.summary()

history = encoder_with_projection_head.fit_generator(
        generator=training_generator,
        steps_per_epoch=training_generator.n / batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n / batch_size,
        verbose=1
)

# save the encoder
encoder.save(encoder_path)


"""
### 3. Train the classifier with the frozen encoder
"""

classifier = create_classifier(encoder, trainable=False)

history = classifier.fit_generator(
        generator=training_generator,
        steps_per_epoch=training_generator.n / batch_size,
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n / batch_size,
        verbose=1
)

plot_history(history)
