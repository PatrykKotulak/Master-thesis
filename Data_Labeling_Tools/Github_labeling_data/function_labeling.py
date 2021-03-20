#! python3

import json
import os
from pathlib import Path


def starting_program():
    '''Function display information when program starts and wait for client
    reaction.'''
    input('Welcome to program for manualy labeling data.\nYou should enter '
          '0 - non-harmful, 1 - offensive language or 2 - punishable threats for displayed '
          'comment. When you are ready, press ENTER')


def create_file_and_header_in_file(filename: str):
    '''Function which create file and header to prepare it for labeled data'''
    if not os.path.isfile(filename):
        with open(filename, 'a', encoding='utf-8') as file_txt:
            file_txt.write('Comment|No|Yes|Kind of offensive language\n')


def create_folder_to_labeled_file(name_folder: str):
    '''Function to create folder for file that will be saved in .txt.
    Args:
        name_folder (str)
    '''
    Path(name_folder).mkdir(parents=True, exist_ok=True)


def open_json(name_file: str):
    '''Open JSON file to display for client
    Args:
        name_file (str): name file to open
    Return:
        comment - opened file
    '''
    with open(name_file.name, 'r', encoding='utf-8') as file_json:
        comment = json.load(file_json)
        print(f'\nKomentarz: {comment["komentarz"]}')
        return comment


def label_for_comment():
    ''' Function to receiving label number for client
    Returns:
        label (int) - number entered by client
    '''
    while True:
        try:
            label = input('Enter your label (if you want to exit, enter y): ')
            if label.lower() == 'y':
                break
            else:
                label = int(label)
                if label in [0, 1, 2]:
                    break
        except:
            pass
        print('Incorrect input, try again. Enter number 0, 1, 2 or y.')
    return label


def open_write_labeled_data(comment: dict, label: int, filename: str):
    '''Function to write comment with label to file
    Args:
        comment (dict json) - received comment
        label (int) - number from client
        filename (str with extension .txt) - file to write labeled data
    '''
    comment['komentarz'] = comment['komentarz'].replace('\r\n', ' ').replace(
	'\n', ' ').replace('|', ' ')
    with open(filename, 'a', encoding='utf-8') as file_txt:
        file_txt.write(f'{comment["komentarz"]}|{comment["nie"]}|'
                       f'{comment["tak"]}|{label}\n')


def change_folder_labeled_file(name_file: str, directory: str):
    '''Function to moved file that has been processed
    Args:
        name_file (str) - name file to transferred
        directory (str) - directiory where file is located
    '''
    os.rename(directory + rf'\{name_file}',
              directory + rf'\Labeled_file\{name_file}')
