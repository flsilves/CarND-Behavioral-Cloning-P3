import os
import csv

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
