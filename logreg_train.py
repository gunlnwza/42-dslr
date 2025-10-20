#!./venv/bin/python

import argparse
import sys

import pandas as pd

from utils.parse_utils import COLUMN_NAMES

# NOTE: feel free to change anything: parsing, function signatures, ...


# TODO: implement with real logistic regression
def ft_fit(X: pd.DataFrame, y: pd.DataFrame):
    model = None

    return model


def main():
    parser = argparse.ArgumentParser(
        prog="logreg_predict.py",
        description="Predict which house the students belong to"
    )
    parser.add_argument("path", help=".csv file to predict")
    parser.add_argument("-s", "--sgd", action="store_true", help="Toggle stochastic gradient descent")
    parser.add_argument("-o", "--optimizer", help="Optimization algorithm to use")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.path, index_col="Index")
        # TODO: validate that it got all the numeric columns, and 'Hogwarts House' target,
        # TODO: (what should we do with 'First Name', 'Last Name', 'Birthday', and 'Best Hand')
    except Exception as e:
        print(e)
        sys.exit(1)

    y = df["Hogwarts House"]
    X = df[COLUMN_NAMES]
    model = ft_fit(X, y)  # train
    # save model


if __name__ == "__main__":
    main()
