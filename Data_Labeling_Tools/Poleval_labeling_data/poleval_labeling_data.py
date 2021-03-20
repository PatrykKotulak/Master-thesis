#!python3
import os
import sys
import function_poleval_labeling as fpl

if __name__ == '__main__':
    name_file_main = 'conected_poleval_comment.txt'
    directory = r'PATH\TO\COMMENTS\TXT'
    name_new_file = 'converted_label_poleval.txt'
    os.chdir(directory)
    fpl.create_file_and_header_in_file(name_new_file)
    last_line = fpl.checking_last_line(name_file_main)
    fpl.starting_program()

    with open(name_file_main, 'r', encoding='utf-8') as file:
        for line in file:
            if (line[-2] == '0' and line != last_line) or \
                    (line[-1] == '0' and line == last_line):
                fpl.open_write_labeled_data_when_zero(name_new_file, line)
            else:
                print(f'\nKomentarz: {line[:-3]}')
                label_comment = fpl.label_for_comment()
                if label_comment == 'y':
                    sys.exit()
                fpl.open_write_labeled_data_when_one(name_new_file, line,
                                                     label_comment)
