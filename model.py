from keras.layers import Flatten, Dense
from keras.models import Sequential
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Cropping2D


class Parameters:
    original_image_dimensions{'width': 320, 'height': 160}


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


def batch_generator(entries, batch_size=32, mirror_image=False):
    num_entries = len(entries)
    while 1:
        shuffle(entries)
        for offset in range(start=0, stop=num_entries, step=batch_size):
            batch_entries = entries[offset: offset + batch_size]

            x_train = []
            y_train = []

            for entry in batch_entries:

                image = cv2.imread(entry.filename_center)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                steering = entry.steering

                if mirror_image:
                    # TODO Check keras tools for data augmentation
                    image = cv2.flip(image, 1)
                    steering = -1.0 * steering

                x_train.append(image)
                y_train.append(steering)

            x_train = np.array(x_train)
            y_train = np.array(y_train)

        batch = shuffle(x_train, y_train)
        yield batch


if __name__ == "__main__":

    entries = read_dataset_entries('data/')

    print(str(entries[0]))
    print(len(entries))

    x_train, y_train = load_images_and_steering(entries)

    print('Size of loaded images: {:d} MB'.format(
        sys.getsizeof(x_train) // 2**20))


    # model
    model = Sequential()
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))

    model.add(Flatten(input_shape=(90, 320, 3)))
    model.add(Dense(1))
	
    print('Compiling model...')
    model.compile(loss='mse', optimizer='adam')
    print('Model fit...')
    model.fit(x_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7, verbose=1)
    model.save('models/data.h5')
