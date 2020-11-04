
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


class Parameters:
    STEERING_CORRECTION = 0.2
    CROPPING_DIMS = ((70, 25), (0, 0))
    INPUT_SHAPE = (160, 320, 3)


class LogEntry:
    """
    Holds the cell values corresponding to a single row on 'driving_log.csv'
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
        image_paths = (self.filename_left,
                       self.filename_center, self.filename_right)
        steering_values = (self.steering + Parameters.STEERING_CORRECTION,
                           self.steering, self.steering - Parameters.STEERING_CORRECTION)

        return list(zip(image_paths, steering_values))


def read_dataset_entries(data_folder_path='data/', skip_header=True):
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


def load_images_and_steering(log_entries):
    """

    """
    images = []
    steering_values = []
    for entry in log_entries:
        image = cv2.imread(entry.filename_center)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
        steering_values.append(entry.steering)
    return (np.array(images), np.array(steering_values))


def batch_generator(samples, batch_size=32, mirror_image=False):
    num_samples = len(entries)
    while 1:
        shuffle(entries)
        print('generator looped through all provided samples')
        for offset in range(0, num_samples, batch_size):
            current_batch = samples[offset: offset + batch_size]

            x_train = []
            y_train = []

            for single_sample in current_batch:

                image_path = single_sample[0]
                steering = single_sample[1]

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if mirror_image:
                    image = cv2.flip(image, 1)
                    steering = -1.0 * steering

                x_train.append(image)
                y_train.append(steering)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

            yield shuffle(x_train, y_train)


class BaseModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: (x / 255.0) -
                              0.5, input_shape=(160, 320, 3)))
        self.model.add(Cropping2D(cropping=Parameters.CROPPING_DIMS,
                                  input_shape=Parameters.INPUT_SHAPE))


class NvidiaModel(BaseModel):
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
    """ Return a list of samples [(image_filepath, steering_value), ...] based on driving_log.csv """
    """ Each entry or row gives three samples (left, center, right) cameras """
    csv_log_entries = read_dataset_entries(data_path)
    samples = []

    for entry in csv_log_entries:
        samples.extend(entry.get_samples())

    return samples


if __name__ == "__main__":

    samples = read_samples_from_csv('data/')

    train_samples, validation_samples = train_test_split(
        samples, test_size=0.3
    )

    print('Number of train entries {:d}'.format(len(train_entries)))
    print('Number of validation entries {:d}'.format(len(validation_entries)))

    training_data_generator = batch_generator(train_entries)
    validation_data_generator = batch_generator(validation_entries)

    # model
    nvidia = NvidiaModel()

    BATCH_SIZE = 32

    training_batches = ceil(len(train_entries) / BATCH_SIZE)
    validation_batches = ceil(len(validation_entries) / BATCH_SIZE)

    nvidia.model.fit_generator(
        training_data_generator,
        validation_data=validation_data_generator,
        steps_per_epoch=training_batches,
        validation_steps=validation_batches,
        epochs=10,
        verbose=1,
    )

    nvidia.model.save('model.h5') """
