import string
from tensorflow import keras
import pathlib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D
from keras.layers import BatchNormalization
import numpy as np
import os
import tensorflow as tf
from keras import Model


def produce_model(train_dataset_path: pathlib.Path, device_name: string, single_model: bool, batch_size=32,
                  img_height=20, img_width=20):
    num_classes = 2

    with tf.device(device_name):
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dataset_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dataset_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def __create_model_cnn():
        model = Sequential([
            keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.AveragePooling2D(),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.AveragePooling2D(),
            keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            keras.layers.AveragePooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(num_classes, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    def __create_model_cnn_2():
        model = Sequential()
        model.add(keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3), name='rescaling_2_1'))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_height, img_width)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(5, 5), strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Conv2D(128, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(num_classes, activation="sigmoid"))
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
        return model

    def __create_LeNet5():
        model = Sequential()
        model.add(keras.layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3), name='rescaling_3_1'))
        model.add(
            Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(img_height, img_width),
                   padding="same"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dense(84, activation='relu'))
        model.add(Dense(num_classes, activation='sigmoid'))
        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
        return model

    def __train_model(model, epochs):
        with tf.device(device_name):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs)
            return history

    model_cnn = __create_model_cnn()
    model_cnn_2 = __create_model_cnn_2()
    model_LeNet = __create_LeNet5()

    __train_model(model_cnn, 30)
    __train_model(model_cnn_2, 30)
    __train_model(model_LeNet, 30)

    if not single_model:
        return model_cnn, model_cnn_2, model_LeNet
    else:
        models = (model_cnn, model_cnn_2, model_LeNet)
        input = keras.layers.Input(shape=(img_height, img_width, 3), name='input')  # input layer
        outputs = [model(input) for model in models]
        x = keras.layers.Average()(outputs)
        conc_model = keras.Model(input, x, name='Concatenated_Model')
        return conc_model


def predict_by_models(test_dataset_path: pathlib.Path, models: (Sequential, Sequential, Sequential), img_height=20, img_width=20):
    detected_points = []
    for path in os.listdir(test_dataset_path):
        img_path = os.path.join(test_dataset_path, path)
        with tf.device('/device:GPU:0'):
            img = tf.keras.utils.load_img(
                img_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions_cnn = np.array(models[0](img_array)),
            predictions_cnn_2 = np.array(models[1](img_array)),
            predictions_LeNet = np.array(models[2](img_array)),
            predictions = (predictions_cnn[0] + predictions_cnn_2[0] + predictions_LeNet[0]) / 3

            score = tf.nn.softmax(predictions[0])

            if np.argmax(score) == 1:
                path_splited = path.split('_')
                detected_points.append((int(path_splited[1]), int(path_splited[2].split('.')[0])))

    return detected_points


def predict_by_single_model(test_dataset_path: pathlib.Path, model, img_height=20, img_width=20):
    detected_points = []
    for path in os.listdir(test_dataset_path):
        img_path = os.path.join(test_dataset_path, path)
        with tf.device('/device:GPU:0'):
            img = tf.keras.utils.load_img(
                img_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            score = tf.nn.softmax(np.array(model(img_array))[0])

            if np.argmax(score) == 1:
                path_splited = path.split('_')
                detected_points.append((int(path_splited[1]), int(path_splited[2].split('.')[0])))

    return detected_points
