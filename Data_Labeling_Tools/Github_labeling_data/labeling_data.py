#! python3

import json
import os
import sys
from pathlib import Path

import function_labeling as fl


if __name__ == '__main__':

    directory = r'PATH\TO\COMMENTS\JSON'
    os.chdir(directory)
    fl.create_folder_to_labeled_file('Labeled_file')
    fl.create_file_and_header_in_file('labeled_dataset.txt')
    fl.starting_program()

    # for loop for all file .json in folder from directory variable
    for name_file in os.scandir(path='.'):
        if name_file.name.endswith(".json") and name_file.is_file():
            comment = fl.open_json(name_file)
            label_comment = fl.label_for_comment()
            if label_comment == 'y':
                sys.exit()
            fl.open_write_labeled_data(comment, label_comment,
                                       'labeled_dataset.txt')
            fl.change_folder_labeled_file(name_file.name, directory)
