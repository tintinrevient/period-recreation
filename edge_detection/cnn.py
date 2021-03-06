from keras import models
from keras import layers
from keras import regularizers
from keras_preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix


# number of training epochs
epochs = 50
# batch size
batch_size = 25
# number of classes
num_classes = 2
# target size
target_size = [96, 96]

# color channel
# rgb
# color_channel = 3
# color_model = 'rgb'

# grayscale
color_channel = 1
color_model = 'grayscale'

# path to the dataset
dataset_train_dir_path = os.path.join('data', 'train')
dataset_test_dir_path = os.path.join('data', 'test')

# path to the trained models
model_dirname = 'model'
if not os.path.exists(model_dirname):
    os.mkdir(model_dirname)

model_filename = 'cnn_' + color_model + '_model.h5'
model_path = os.path.join(model_dirname, model_filename)

# path to the trained models' weights
weights_dirname = 'weights'
if not os.path.exists(weights_dirname):
    os.mkdir(weights_dirname)

weights_filename = 'weights_' + color_model + '_model.npy'
weights_path = os.path.join(weights_dirname, weights_filename)

def preprocess(testgen_class_mode="categorical"):

    # class_mode
    # "categorical" will be 2D one-hot encoded labels
    # "binary" will be 1D binary labels
    # "sparse" will be 1D integer labels
    # "input" will be images identical to input images (mainly used to work with autoencoders)

    train_data_generator = image.ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=[0.9, 1.4],
        brightness_range=[0.75, 1.25],
        rotation_range=10
    )

    test_data_generator = image.ImageDataGenerator(
        rescale=1. / 255
    )

    training_generator = train_data_generator.flow_from_directory(
        directory=dataset_train_dir_path,
        target_size=tuple(target_size),
        color_mode= color_model,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    validation_generator = train_data_generator.flow_from_directory(
        directory=dataset_train_dir_path,
        target_size=tuple(target_size),
        color_mode= color_model,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    test_generator = test_data_generator.flow_from_directory(
        directory=dataset_test_dir_path,
        target_size=tuple(target_size),
        color_mode= color_model,
        batch_size=batch_size,
        class_mode=testgen_class_mode
    )

    return training_generator, validation_generator, test_generator


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


def plot_confusion_matrix(model_path, test_generator):

    model = load_model(model_path)

    total_images = test_generator.n
    steps = total_images // batch_size

    x_test, y_test = [], []
    for step in range(steps):
        x_one_batch, y_one_batch = test_generator.next()

        x_test.extend(x_one_batch)
        y_test.extend(y_one_batch)

    print('X test size:', len(x_test))
    print('Y test size:', len(y_test))

    # [] to np.array
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_true = y_test
    y_pred = model.predict_classes(x_test)

    print('Y true size:', len(y_true))
    print('Y pred size:', len(y_pred))

    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    plt.show()


def train():
    training_generator, validation_generator, test_generator = preprocess()

    model = models.Sequential([
        # conv 1
        layers.SeparableConv2D(filters=32,
                               kernel_size=(5, 5),
                               kernel_regularizer=regularizers.l2(0.001),
                               activation='relu',
                               input_shape=tuple(target_size) + (color_channel,)),
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

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit_generator(
        generator=training_generator,
        steps_per_epoch=training_generator.n / batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n / batch_size,
        verbose=1
    )

    plot_history(history)

    # evaluate by test dataset
    loss, accuracy = model.evaluate_generator(test_generator)
    print('\nTest accuracy:', accuracy)

    # save the model
    model.save(model_path)

    # save the model's weights
    np.save(weights_path, model.get_weights())


def predict(genre, img_filename, color_model):

    datagen = image.ImageDataGenerator(
        rescale=1. / 255
    )

    testgen = datagen.flow_from_directory(
        directory=dataset_test_dir_path,
        target_size=tuple(target_size),
        color_mode=color_model,
        batch_size=batch_size,
        class_mode="categorical"
    )

    # class indices dictionary with the mapping: class_name -> class_index
    class_indices = testgen.class_indices
    class_names = []
    for i, class_name in enumerate(class_indices):
        class_names.append(class_name)

    print('Class indices:', class_indices)

    # load the test image
    img_path = os.path.join(dataset_test_dir_path, genre, img_filename)
    img = image.load_img(img_path, target_size=tuple(target_size), color_mode=color_model)
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array = img_array / 255.

    # load the model and predict the label
    model = models.load_model(model_path)
    pred = model.predict(img_array)

    predicted_label = class_names[np.argmax(pred)]
    predicted_prob = np.max(pred)

    true_label = genre

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
    # train()

    # plot the confusion matrix
    # training_generator, validation_generator, test_generator = preprocess(testgen_class_mode="sparse")
    # plot_confusion_matrix(model_path=model_path, test_generator=test_generator)

    # predict the label given the image index
    predict('landscape', '0a8f3033e279b8a7123ae7fe629b611dc.jpg', color_model) # 100% landscape
