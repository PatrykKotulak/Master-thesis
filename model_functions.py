#! python3

import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from argparse import Namespace
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc, log_loss


def split_train_val_test(train_percent=0.6,
                         val_percent=0.2,
                         test_percent=0.2):
    args = Namespace(
        dataset_csv="data_for_model.csv",
        train_percent=train_percent,
        val_percent=val_percent,
        test_percent=test_percent,
        output_csv="data_with_splits.csv",
        seed=101
    )

    # Read raw data
    comments = pd.read_csv(args.dataset_csv, header=0)

    # Splitting train by nationality
    # Create dict
    by_kind_language = collections.defaultdict(list)
    for _, row in comments.iterrows():
        by_kind_language[row['Kind of offensive language']].append(
            row.to_dict())

    # Create split data
    ready_list = []
    np.random.seed(args.seed)
    for _, item_lists in sorted(by_kind_language.items()):
        np.random.shuffle(item_lists)
        n = len(item_lists)
        n_train = int(args.train_percent * n)
        n_val = int(args.val_percent * n)
        n_test = int(args.test_percent * n)

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
    ready_comments_split = pd.DataFrame(ready_list)

    return ready_comments_split
    # Write munged data to CSV
    # ready_comments_split.to_csv(args.output_csv, index=False)


def train_validation_test_split(data,
                                col_stratify='Kind of offensive language',
                                train_percent=0.6,
                                validate_percent=0.2,
                                test_percent=0.2,
                                random_state=101):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test).
    Function uses train_test_split (from sklearn) and stratify to receive
    the same ratio response (y, target) in each splits.

    Args
        data (dataframe)
        col_stratify (str): name of column target
        train_percent (float)
        validate_percent (float)
        test_percent (float)
        random_state (int, None)
    Sum of train_percent, validate_percent and test_percent have to be
    equal 1.0.

    Returns
        data_train, data_val, data_test : Dataframes containing the three splits.
    '''

    if train_percent + validate_percent + test_percent != 1.0:
        raise ValueError(f'Sum of train, validate and test is not 1.0')

    if col_stratify not in data.columns:
        raise ValueError(f'{col_stratify} is not a column in the dataframe')

    X = data
    y = data[[col_stratify]]

    # Split original dataframe into train and temp dataframes.
    data_train, data_temp, y_train, y_temp = train_test_split(X,
                                                              y,
                                                              stratify=y,
                                                              test_size=(
                                                                      1.0 - train_percent),
                                                              random_state=random_state)
    # Split the temp dataframe into val and test dataframes.
    test_to_split = test_percent / (validate_percent + test_percent)
    data_val, data_test, y_validate, y_test = train_test_split(data_temp,
                                                                    y_temp,
                                                                    stratify=y_temp,
                                                                    test_size=test_to_split,
                                                                    random_state=random_state)

    assert len(data) == len(data_train) + len(data_val) + len(data_test)

    # Connect split data
    data_train['split'] = 'train'
    data_val['split'] = 'val'
    data_test['split'] = 'test'
    data = pd.concat([data_train, data_val, data_test])
    data.reset_index(drop=True, inplace=True)

    return data


result = ''
the_best_result = ''


class Modeling:
    """Modeling and presentation of results"""

    def __init__(self, model, X_train, X_test, title):
        """Inicjalization"""

        self.model = model
        self.title = title
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.result = ''

    def fit_predict(self):
        """Function to train, predict our model and create roc curve"""
        self.classifier = self.model
        self.classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.classifier.predict(self.X_test)
        self.y_pred_proba = self.classifier.predict_proba(self.X_test)[:, 1]

        self.y_train_proba = self.classifier.predict_proba(self.X_train)[:, 1]

        self.fpr_train, self.tpr_train, self.thr_train = roc_curve(
            self.y_train, self.y_train_proba)
        self.fpr, self.tpr, self.thr = roc_curve(y_test, self.y_pred_proba)

    def print_results(self):
        """Function to print our result"""
        self.accuracy = round(accuracy_score(y_test, self.y_pred), 4)
        self.f1 = round(f1_score(y_test, self.y_pred), 4)
        self.recall = round(recall_score(y_test, self.y_pred), 4)
        self.log_loss = round(log_loss(y_test, self.y_pred_proba), 4)

        print(f'Results for {self.title}:')
        print(f'{self.title} accuracy: {self.accuracy}')
        print(f'{self.title} f-score: {self.f1}')
        print(f'{self.title} recall: {self.recall}')
        print(f'{self.title} log_loss: {self.log_loss}')

    def add_to_table(self):
        """Function to add our result to dataframe to compare all"""
        global result
        if len(result) == 0:
            result = {self.title: [self.accuracy, self.f1, self.recall,
                                   self.log_loss]}
            result = pd.DataFrame(result, index=['Accuracy', 'F-score',
                                                 'Recall', 'Log_loss'])
        else:
            conact = {self.title: [self.accuracy, self.f1, self.recall,
                                   self.log_loss]}
            conact = pd.DataFrame(conact, index=['Accuracy', 'F-score',
                                                 'Recall', 'Log_loss'])
            result = pd.concat([result, conact], axis=1)

    def plot_confusion_matrix(self):
        """plot confusion matrix"""
        plt.figure(figsize=(10, 10), facecolor='w')
        sns.heatmap(confusion_matrix(y_test, self.y_pred), annot=True,
                    fmt='.0f',
                    cbar=False,
                    vmax=confusion_matrix(y_test, self.y_pred).max(),
                    vmin=0, cmap='Blues')
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion matrix for {self.title}')

    def plot_confusion_matrix_percent(self):
        """Plot confusion matrix with part of 1 value"""
        plt.figure(figsize=(10, 10), facecolor='w')
        cm = confusion_matrix(y_test, self.y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(cm_norm)
        sns.heatmap(df_cm, annot=True, cmap="Blues", cbar=False)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title(f'Confusion matrix for {self.title}')