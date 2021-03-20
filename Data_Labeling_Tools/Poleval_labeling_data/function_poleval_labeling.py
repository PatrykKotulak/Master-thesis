#!python3

import os
import sys


def starting_program():
    '''Function display information when program starts and wait for client
    reaction.'''
    input('Welcome to program for manualy labeling data.\nYou should enter '
          '1 - offensive language or 2 - punishable threats for displayed '
          'comment. When you are ready, press ENTER')


def create_file_and_header_in_file(filename: str):
    '''Function which create file and header to prepare it for labeled data'''
    with open(filename, 'w', encoding='utf-8') as file_txt:
        file_txt.write('Comment,Kind of offensive language\n')


def checking_last_line(name_file):
    '''Function to receive last line of file
    Args:
        name_file (str)
    Returns:
        last_line (str)
        '''
    with open(name_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        return last_line


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
                if label in [1, 2]:
                    break
        except:
            pass
        print('Incorrect input, try again. Enter number 1, 2 or y.')
    return label


def open_write_labeled_data_when_zero(filename: str, line: str):
    '''Function to write comment with label to file
    Args:
        filename (str with extension .txt) - file to write labeled data
        line (int) - line of file
    '''
    with open(filename, 'a', encoding='utf-8') as file_txt:
        file_txt.write(line)


def open_write_labeled_data_when_one(filename: str, line: str, label: int):
    '''Function to write comment with label to file
    Args:
        filename (str with extension .txt) - file to write labeled data
        line (int) - line of file
        label (int) - number from client
    '''
    with open(filename, 'a', encoding='utf-8') as file_txt:
        file_txt.write(f'{line[:-2]}{label}\n')
