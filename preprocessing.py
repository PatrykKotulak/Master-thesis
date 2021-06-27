import re
import collections
import emoji
import pandas as pd
import numpy as np
import torch

# processing of text
import spacy
from autocorrect import Speller
from stopwords import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# load and download
spell = Speller('pl')
lemma_spacy = spacy.load('pl_spacy_model')


class Preprocessing:
    def __init__(self):
        self.conected_data = None
        self.x_raw = None
        self.y = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.dict_words_place = collections.defaultdict(list)

    def load_data(self):
        data_poleval_raw = pd.read_csv('Labeled_data/converted_label_poleval.txt', error_bad_lines=False, sep=",")
        data_poleval_test_raw = pd.read_csv('Labeled_data/converted_label_poleval_test.txt',
                                            error_bad_lines=False, sep=",")
        data_github_raw = pd.read_csv('Labeled_data/labeled_dataset.txt', error_bad_lines=False, sep="|")
        data_poleval = data_poleval_raw.copy(deep=True)
        data_poleval_test = data_poleval_test_raw.copy(deep=True)
        data_github = data_github_raw.copy(deep=True)

        self.conected_data = pd.concat([data_poleval,
                                        data_poleval_test,
                                        data_github.drop(['No', 'Yes'], axis=1)], axis=0, ignore_index=True)

    def remove_quoting_comments(self):
        pattern = r'^RT.*'
        remove = self.conected_data['Comment'].str.contains(pattern)
        data = self.conected_data[~remove].reset_index(drop=True)
        self.x_raw = data['Comment']
        self.y = data['Kind of offensive language']

    def demojize(self):
        self.x_raw = self.x_raw.apply(
            lambda x: emoji.demojize(x, delimiters=("~~", "~~")))

    def clean_text(self):
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
            lambda x: [item for item in x if item not in stopwords and len(item) >= 2])
        for i in range(len(self.x_raw)):
            self.x_raw[i] = ' '.join(self.x_raw[i])

    def lemmatize_text(self):
        self.x_raw = self.x_raw.apply(lambda x: [token.lemma_ for token in lemma_spacy(x)])

    def correct_typo_words(self):
        for _, row_list in self.x_raw.items():
            for place_in_row in range(len(row_list)):
                self.dict_words_place[row_list[place_in_row]].append((_, place_in_row))

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
        self.file_write = pd.concat([self.x_raw, self.y], axis=1)
        self.file_write.replace('', np.nan, inplace=True)
        self.file_write.dropna(inplace=True)

    def write_to_file(self, name='cleaned_data'):
        self.file_write.to_csv(f'{name}.csv', index=False)

    def split_train_val_test(self, train_percent=0.6,
                             val_percent=0.2,
                             test_percent=0.2,
                             visualization=False,
                             file_name='cleaned_data'):
        # Read raw data
        comments = pd.read_csv(f'{file_name}.csv', header=0)

        # Splitting train by nationality
        # Create dict
        by_kind_language = collections.defaultdict(list)
        for _, row in comments.iterrows():
            by_kind_language[row['Kind of offensive language']].append(
                row.to_dict())

        # Create split data
        ready_list = []
        np.random.seed(101)
        for _, item_lists in sorted(by_kind_language.items()):
            np.random.shuffle(item_lists)
            n = len(item_lists)
            n_train = int(train_percent * n)
            n_val = int(val_percent * n)
            n_test = int(test_percent * n)

            # Give data point a split attribute
            for item in item_lists[:n_train]:
                item['split'] = 'train'
            for item in item_lists[n_train:n_train + n_val]:
                item['split'] = 'val'
            for item in item_lists[n_train + n_val:]:
                item['split'] = 'test'

                # Add to final list
            ready_list.extend(item_lists)

        # Write split data to file
        split_comments = pd.DataFrame(ready_list)
        if not visualization:
            self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
                split_comments[split_comments['split'] == 'train']['Comment'], \
                split_comments[split_comments['split'] == 'val']['Comment'], \
                split_comments[split_comments['split'] == 'test']['Comment'], \
                split_comments[split_comments['split'] == 'train']['Kind of offensive language'], \
                split_comments[split_comments['split'] == 'val']['Kind of offensive language'], \
                split_comments[split_comments['split'] == 'test']['Kind of offensive language']
        else:
            return split_comments

    def count_vectorizer(self):
        vectorizer = CountVectorizer()
        self.X_train_cv = vectorizer.fit_transform(self.X_train.astype('U').values)
        self.X_val_cv = vectorizer.transform(self.X_val.astype('U').values)
        self.X_test_cv = vectorizer.transform(self.X_test.astype('U').values)

    def tfidf_vectorizer(self):
        tfidf_vectorizer = TfidfVectorizer()
        self.X_train_tfidf = tfidf_vectorizer.fit_transform(self.X_train.astype('U').values)
        self.X_val_tfidf = tfidf_vectorizer.transform(self.X_val.astype('U').values)
        self.X_test_tfidf = tfidf_vectorizer.transform(self.X_test.astype('U').values)

    def sparse_to_tensor(self, X_train, X_val, X_test):
        self.X_train_tensor = torch.from_numpy(X_train.todense()).float()
        self.X_val_tensor = torch.from_numpy(X_val.todense()).float()
        self.X_test_tensor = torch.from_numpy(X_test.todense()).float()
        self.y_train_tensor = torch.from_numpy(np.array(self.y_train))
        self.y_val_tensor = torch.from_numpy(np.array(self.y_val))
        self.y_test_tensor = torch.from_numpy(np.array(self.y_test))