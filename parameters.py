""" Parameters """


class Parameters:
    BATCH_SIZE = 32
    EPOCHS = 3
    TEST_SIZE_FRACTION = 0.2

    # Input shape of the original camera images (height, width, color layers)
    INPUT_SHAPE = (160, 320, 3)

    # Steering correction value for left (+) and right (-) images
    STEERING_CORRECTION = 0.2

    # Pixels to crop in pre-processing: ((top, bottom),(left,right))
    CROPPING_DIMENSIONS = ((70, 25), (0, 0))

    DATASET_FOLDERS = [
        # "data/provided/",
        "data/track1_center/",
        "data/track1_reverse/",
        "data/track1_right/",
        "data/track1_left/",
        "data/track2_center/",
        "data/track2_reverse/",
    ]
