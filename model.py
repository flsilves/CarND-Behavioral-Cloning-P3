
import numpy as np
import matplotlib.pyplot as plt

from math import ceil

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from parameters import Parameters
from data_reader import read_dataset_entries


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