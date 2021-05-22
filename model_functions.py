from sklearn.model_selection import train_test_split


def train_validation_test_split(data,
                                col_stratify='Kind of offensive language',
                                train_percent=0.6,
                                validate_percent=0.2,
                                test_percent=0.2,
                                random_state=None):
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
        data_train, data_validate, data_test : Dataframes containing the three splits.
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
    data_validate, data_test, y_validate, y_test = train_test_split(data_temp,
                                                                    y_temp,
                                                                    stratify=y_temp,
                                                                    test_size=test_to_split,
                                                                    random_state=random_state)

    assert len(data) == len(data_train) + len(data_validate) + len(data_test)

    return data_train, data_validate, data_test


