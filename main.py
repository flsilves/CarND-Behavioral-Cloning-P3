""" Main Module """

import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2

from math import ceil
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from parameters import Parameters
from models import NvidiaModel
from data_reader import read_samples_from_csv


def batch_generator(samples, batch_size=Parameters.BATCH_SIZE):
    """ 
    Batch generator with data augmentation by mirroring the image horizontally 
    Each sample produces two images, the step is hence 'batch_size//2' 

    """
    assert batch_size % 2 == 0, 'batch_size should be an even number'

    num_samples = len(samples)
    while 1:
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

        logging.debug(
            'Generator looped through all provided samples, shuffling...')


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')

    samples = []

    for data_folder in Parameters.DATASET_FOLDERS:
        samples.extend(read_samples_from_csv(data_folder))

    logging.info('size of data {:d}'.format(len(samples)))

    train_samples, validation_samples = train_test_split(
        samples, test_size=Parameters.TEST_SIZE_PERCENTAGE
    )

    logging.info('Number of train samples {:d}'.format(len(train_samples)))
    logging.info('Number of validation samples {:d}'.format(
        len(validation_samples)))

    batch_size = Parameters.BATCH_SIZE

    training_data_generator = batch_generator(
        train_samples, batch_size)
    validation_data_generator = batch_generator(
        validation_samples, batch_size)

    nvidia = NvidiaModel()

    # Each sample produces 2 samples because of data augmentation in the generator (horizonal mirror)
    #
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
