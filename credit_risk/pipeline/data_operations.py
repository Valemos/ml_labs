from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
import numpy as np


def get_data_xy(df, y_columns):
    return df[df.columns.difference(y_columns)], df[y_columns]


def split_train_data(df, y_columns, random_seed, test_size=0.2):
    return split_train_xy(*get_data_xy(df, y_columns), random_seed, test_size)


def split_train_xy(x, y, random_seed, test_size):
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=random_seed)


def f1_score(y_true, y_pred):

    y_true = np.where(np.asarray(y_true) < 0.5, 0, 1).reshape((y_true.shape[0],))
    y_pred = np.where(np.asarray(y_pred) < 0.5, 0, 1).reshape((y_pred.shape[0],))

    # Count positive samples.
    true_positive = np.sum(y_pred * y_true)
    false_positive = combine_predictions(y_pred, y_true, 1, 0)
    false_negative = combine_predictions(y_pred, y_true, 0, 1)
    true_negative = combine_predictions(y_pred, y_true, 0, 0)

    print(f"true positive: {true_positive}")
    print(f"true negative: {true_negative}")
    print(f"false positive: {false_positive}")
    print(f"false negative: {false_negative}")

    # If there are no true samples, fix the F1 score at 0.
    if true_positive + false_positive == 0:
        return 0

    # How many selected items are relevant?
    precision = true_positive / (true_positive + false_positive)

    # How many relevant items are selected?
    recall = true_positive / (true_positive + false_negative)

    # Calculate f1_score
    score = 2 * (precision * recall) / (precision + recall)
    return score


def combine_predictions(y_pred, y_true,value_predicted, value_true):
    return np.sum(np.where(y_pred == value_predicted, 1, 0) * np.where(y_true == value_true, 1, 0))


def show_f1_score(model, x, y):
    print("f1 score:", f1_score(np.asarray(y), model.predict(x)))


def cross_validate(model, x, y, n_splits=10, n_repeats=1, random_state=1):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    scores = cross_val_score(model, x, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    return np.mean(scores), scores

def print_dataframe_diff(df1, df2):
    print(df1.columns.difference(df2.columns))
    print(df2.columns.difference(df1.columns))
