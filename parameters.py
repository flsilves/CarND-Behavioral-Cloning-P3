""" Parameters """


class Parameters:
    BATCH_SIZE = 32
    EPOCHS = 3

    TEST_SIZE_PERCENTAGE = 0.3

    DATASET_FOLDERS = [
        "data/provided/",
        "data/track1_center/",
        "data/track1_reverse/",
        "data/track1_right/",
        "data/track1_left/",
        "data/track2_center/",
        "data/track2_reverse/",
    ]

    INPUT_SHAPE = (160, 320, 3)
    STEERING_CORRECTION = 0.2
    CROPPING_DIMS = ((70, 25), (0, 0))
