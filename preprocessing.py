import re
import collections
import emoji
import pandas as pd

# processing of text
import spacy
from autocorrect import Speller
from stopwords import stopwords


# load and download
spell = Speller('pl')
lemma_spacy = spacy.load('pl_spacy_model')


class Preprocessing:
    def __init__(self):
        self.x_raw = None
        self.y = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_data(self):
        data_poleval_raw = pd.read_csv(
            'Labeled_data/converted_label_poleval.txt',
            error_bad_lines=False, sep=",")
        data_poleval_test_raw = pd.read_csv(
            'Labeled_data/converted_label_poleval_test.txt',
            error_bad_lines=False, sep=",")
        data_github_raw = pd.read_csv('Labeled_data/labeled_dataset.txt',
                                      error_bad_lines=False, sep="|")

        data_poleval = data_poleval_raw.copy(deep=True)
        data_poleval_test = data_poleval_test_raw.copy(deep=True)
        data_github = data_github_raw.copy(deep=True)

        conected_data = pd.concat([data_poleval,
                                   data_poleval_test,
                                   data_github.drop(['No', 'Yes'],
                                                    axis=1)],
                                  axis=0,
                                  ignore_index=True)
        self.x_raw = conected_data['Comment']
        self.y = conected_data['Kind of offensive language']

    def remove_quoting_comments(self):
        pattern = r'^RT.*'
        remove = self.x_raw.str.contains(pattern)
        self.x_raw = self.x_raw[~remove].reset_index(drop=True)

    def demojize(self):
        self.x_raw = self.x_raw.apply(
            lambda x: emoji.demojize(x, delimiters=("~~", "~~")))

    def clean_text(self, stopwords_remove=False):
        # remove of @name
        pattern = re.compile(r'@\w+[\s]*')
        self.x_raw = self.x_raw.str.replace(pattern, '')

        # split emoji
        pattern = re.compile(r"~{2}")
        self.x_raw = self.x_raw.str.replace(pattern, ' ')

        # remove of links https
        pattern = re.compile(r"https?[:\/\/]+[a-zA-Z0-9.\-\/?=_~:#%]+")
        self.x_raw = self.x_raw.str.replace(pattern, '')

        # removal of punctuations and numbers
        pattern = re.compile(r'[^_ąćęłńóśźżĄĆĘŁŃÓŚŹŻa-zA-Z\s]')
        self.x_raw = self.x_raw.str.replace(pattern, '')

        # remove more than one space
        pattern = re.compile(r'\s+')
        self.x_raw = self.x_raw.str.replace(pattern, ' ')

        # remove beginning and ending task space
        pattern = re.compile(r'^\s+|\s+?$')
        self.x_raw = self.x_raw.str.replace(pattern, '')

        # remove of capitalization
        self.x_raw = self.x_raw.str.lower()

    # remove stopwords
    def stopwords_remove(self):
        self.x_raw = self.x_raw.apply(lambda x: x.split())
        self.x_raw = self.x_raw.apply(
            lambda x: [item for item in x if
                       item not in stopwords and len(item) >= 3])
        for i in range(len(self.x_raw)):
            self.x_raw[i] = ' '.join(self.x_raw[i])

    def lemmatize_text(self):
        self.x_raw = self.x_raw.apply(
            lambda x: [token.lemma_ for token in lemma_spacy(x)])

    def correct_typo_words(self):
        self.dict_words_place = collections.defaultdict(list)
        for _, row_list in self.x_raw.items():
            for place_in_row in range(len(row_list)):
                self.dict_words_place[row_list[place_in_row]].append(
                    (_, place_in_row))

        with open('slowa.txt', encoding='utf-8') as file:
            contents = file.read()
            for word, place_word in self.dict_words_place.items():
                if word in contents:
                    continue
                else:
                    for token in lemma_spacy(spell(word)):
                        correct = token.lemma_
                    for number in place_word:
                        self.x_raw[number[0]][number[1]] = correct

    def token_join(self):
        self.x_raw = self.x_raw.apply(lambda x: ' '.join(x))

    def write_to_file(self):
        self.x_raw.to_csv('clean_x_raw.csv', index=False)
        self.y.to_csv('clean_y.csv', index=False)