
import os
import sys
import cv2
import csv

import numpy as np
import matplotlib.pyplot as plt

from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from parameters import Parameters


class LogEntry:
    """
    Stores the collumn values corresponding to a single row in 'driving_log.csv'
    """

    def __init__(self, line, img_directory_path):
        self.filename_center = os.path.join(
            img_directory_path, line[0].split("/")[-1])
        self.filename_left = os.path.join(
            img_directory_path, line[1].split("/")[-1])
        self.filename_right = os.path.join(
            img_directory_path, line[2].split("/")[-1])
        self.steering = float(line[3])
        self.throttle = float(line[4])
        self.break_value = float(line[5])
        self.speed = float(line[6])

    def __str__(self):
        return ("LogEntry:\n"
                + "\t image_center: '{:s}'\n".format(self.filename_center)
                + "\t image_left: '{:s}'\n".format(self.filename_left)
                + "\t image_right: '{:s}'\n".format(self.filename_right)
                + "\t steering: '{:f}'\n".format(self.steering)
                + "\t throttle: '{:f}'\n".format(self.throttle)
                + "\t break: '{:f}'\n".format(self.break_value)
                + "\t speed: '{:f}'\n".format(self.speed)
                )

    def get_samples(self):
        """ Gets a list of three tuples '(image_filepath, steering)' from the single entry"""
        image_paths = (self.filename_left,
                       self.filename_center, self.filename_right)
        steering_values = (self.steering + Parameters.STEERING_CORRECTION,
                           self.steering, self.steering - Parameters.STEERING_CORRECTION)

        return list(zip(image_paths, steering_values))


def read_dataset_entries(data_folder_path, skip_header=False):
    """
    Reads 'driving_log.csv' and returns a list of LogEntry(s) holding the cell values

    'skip_header': is used to ignore the first line of the csv file

    """

    img_directory_path = os.path.join(data_folder_path, "IMG/")
    csv_file_path = os.path.join(data_folder_path, "driving_log.csv")

    print("Reading entries from: {:s} with skip_header={:b}:".format(
        csv_file_path, skip_header))

    entries = []

    with open(csv_file_path) as csvfile:
        reader = csv.reader(csvfile)
        if skip_header:
            next(reader, None)

        for line in reader:
            entries.append(LogEntry(line, img_directory_path))

    print("Read {:d} entries".format(len(entries)))
    return entries


def batch_generator(samples, batch_size=32):
    """ Batch generator with data augmentation by mirroring the image horizontally """
    """ Each sample produces two images, the step is hence 'batch_size//2' """

    num_samples = len(samples)
    while 1:
        print('Generator looped through all provided samples, shuffling...')
        shuffle(samples)
        for offset in range(0, num_samples, batch_size//2):
            current_batch = samples[offset: offset + batch_size//2]

            x_train = []
            y_train = []

            for single_sample in current_batch:

                image_path = single_sample[0]
                steering = single_sample[1]

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                x_train.append(image)
                y_train.append(steering)

                image = cv2.flip(image, 1)
                steering = -1.0 * steering
                x_train.append(image)
                y_train.append(steering)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            yield shuffle(x_train, y_train)


class BaseModel:
    """ Base model with normalization of image values and image cropping """

    def __init__(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: (x / 255.0) -
                              0.5, input_shape=(160, 320, 3)))
        self.model.add(Cropping2D(cropping=Parameters.CROPPING_DIMS,
                                  input_shape=Parameters.INPUT_SHAPE))


class NvidiaModel(BaseModel):
    """ Nvidia model based on ... """

    def __init__(self):
        BaseModel.__init__(self)
        self.model.add(Convolution2D(
            24, 5, 5, subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(
            36, 5, 5, subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(
            48, 5, 5, subsample=(2, 2), activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(Convolution2D(64, 3, 3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')


def read_samples_from_csv(data_path):
    """ Return a list of samples [(image_filepath, steering_value), ...] from 'driving_log.csv' """
    """ Each entry/row gives produces 3 samples corresponding to the (left, center, right) images """
    csv_log_entries = read_dataset_entries(data_path)
    samples = []

    for entry in csv_log_entries:
        samples.extend(entry.get_samples())

    return samples


if __name__ == "__main__":

    samples = []

    for data_folder in Parameters.DATASET_FOLDERS:
        samples.extend(read_samples_from_csv(data_folder))

    print('size of data {:d}'.format(len(samples)))

    train_samples, validation_samples = train_test_split(
        samples, test_size=0.3
    )

    print('Number of train samples {:d}'.format(len(train_samples)))
    print('Number of validation samples {:d}'.format(len(validation_samples)))

    batch_size = Parameters.BATCH_SIZE

    training_data_generator = batch_generator(
        train_samples, batch_size)
    validation_data_generator = batch_generator(
        validation_samples, batch_size)

    nvidia = NvidiaModel()

    # Each sample produces 2 samples because of data augmentation (horizonal mirror)
    # The number of batches
    training_batches = ceil(2*len(train_samples) / batch_size)
    validation_batches = ceil(2*len(validation_samples) / batch_size)

    nvidia.model.fit_generator(
        training_data_generator,
        validation_data=validation_data_generator,
        steps_per_epoch=training_batches,
        validation_steps=validation_batches,
        epochs=Parameters.EPOCHS,
        verbose=1,
    )

    nvidia.model.save('model.h5')
