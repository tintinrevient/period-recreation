from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras_preprocessing import image
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


# model path
model_file_path = os.path.join('model', 'cnn.h5')
model_weights_path = os.path.join('model', 'cnn.npy')

# dataset path
dataset_train_dir_path = os.path.join('data', 'train')
dataset_test_dir_path = os.path.join('data', 'test')

# hyper-parameters
nb_channel=3
target_size = [112, 112]
epochs = 50
batch_size = 25
learning_rate = 0.001

# number of classes
num_classes = 40


def build_model():

    model = models.Sequential([
        # conv 1
        layers.SeparableConv2D(filters=32,
                               kernel_size=(5, 5),
                               kernel_regularizer=regularizers.l2(0.001),
                               activation='relu',
                               input_shape=tuple(target_size)+(3,)),
        # conv 2
        layers.SeparableConv2D(filters=64,
                               kernel_size=(5, 5),
                               kernel_regularizer=regularizers.l2(0.001),
                               activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        # conv 3
        layers.SeparableConv2D(filters=64,
                               kernel_size=(5, 5),
                               kernel_regularizer=regularizers.l2(0.001),
                               activation='relu'),
        # conv4
        layers.SeparableConv2D(filters=128,
                               kernel_size=(5, 5),
                               kernel_regularizer=regularizers.l2(0.001),
                               activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        # conv 5
        layers.SeparableConv2D(filters=128,
                               kernel_size=(5, 5),
                               kernel_regularizer=regularizers.l2(0.001),
                               activation='relu'),
        # conv 6
        layers.SeparableConv2D(filters=256,
                               kernel_size=(5, 5),
                               kernel_regularizer=regularizers.l2(0.001),
                               activation='relu'),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.BatchNormalization(),

        # flatten
        layers.GlobalAveragePooling2D(),

        # dropout
        # layers.Dropout(0.5),

        # dense
        layers.Dense(units=512,
                     kernel_regularizer=regularizers.l2(0.001),
                     activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(units=num_classes, activation='softmax')
    ])

    return model


def plot_history(history):
    """Plot the training and validation accuracy and loss graphs

    :param history: History of the whole training process
    """
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


def train():

    model = build_model()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # print the summary of model
    model.summary()

    # data augmentation
    train_datagen = image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=[0.9, 1.4],
        brightness_range=[0.75, 1.25],
        rotation_range=10)

    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=dataset_train_dir_path,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    validation_generator = train_datagen.flow_from_directory(
        directory=dataset_train_dir_path,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    test_generator = test_datagen.flow_from_directory(
        directory=dataset_test_dir_path,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode="categorical"
    )

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=train_generator.n / batch_size,
                                  validation_data=validation_generator,
                                  validation_steps=validation_generator.n / batch_size,
                                  epochs=epochs,
                                  verbose=1)

    # plot the training and validation accuracy and loss over the epochs
    plot_history(history)

    # evaluate by the test dataset
    loss, accuracy = model.evaluate_generator(test_generator)
    print('\nTest accuracy:', accuracy)

    # save the model
    model.save(model_file_path)

    # save the model's weights
    np.save(model_weights_path, model.get_weights())


def predict(genre, img_filename):

    datagen = image.ImageDataGenerator(rescale=1. / 255)

    testgen = datagen.flow_from_directory(
        directory=dataset_test_dir_path,
        target_size=(96, 96),
        batch_size=batch_size,
        class_mode="categorical")

    # class indices dictionary with the mapping: class_name -> class_index
    class_indices = testgen.class_indices
    class_names = []
    for i, class_name in enumerate(class_indices):
        class_names.append(class_name)

    # load the test image
    img_path = os.path.join(dataset_test_dir_path, str(genre), img_filename)
    img = image.load_img(img_path, target_size=(96, 96))
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array = img_array / 255.

    # load the model and predict the label
    model = models.load_model(model_file_path)
    pred = model.predict(img_array)

    predicted_label = class_names[np.argmax(pred)]
    predicted_prob = np.max(pred)

    true_label = str(genre)

    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # plot the image
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                         100 * predicted_prob,
                                         true_label), color=color)
    plt.show()


if __name__ == "__main__":

    # train the model
    train()

    # predict the label given the image index
    # predict('landscape', 'mw00001.jpg')  # 60% 1700
