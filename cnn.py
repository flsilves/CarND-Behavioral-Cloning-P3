
""" Convolutional neural network models """

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, ReLU
from keras.layers.pooling import MaxPooling2D

from parameters import Parameters


class BaseModel:
    """ Base model with normalization of image values and image cropping """

    def __init__(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: (x / 255.0) -
                              0.5, input_shape=Parameters.INPUT_SHAPE))
        self.model.add(Cropping2D(cropping=Parameters.CROPPING_DIMENSIONS,
                                  input_shape=Parameters.INPUT_SHAPE))


class NvidiaModel(BaseModel):
    """  Nvidia model based on:
         https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
    """

    def __init__(self):
        BaseModel.__init__(self)
        self.model.add(Convolution2D(
            24, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Convolution2D(
            36, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Convolution2D(
            48, (5, 5), strides=(2, 2), activation='relu'))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(Convolution2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')
