import pandas as pd
from pathlib import Path

_path_data_folder = Path(r"D:\coding\Jupyter_notebooks\ML\credit_risk\data")
_path_train = _path_data_folder / "application_train.csv"
_path_test = _path_data_folder / "application_test.csv"
_path_col_description = _path_data_folder / "HomeCredit_columns_description.csv"


def read_data_train():
    return pd.read_csv(_path_train)

def read_data_test():
    return pd.read_csv(_path_test)
