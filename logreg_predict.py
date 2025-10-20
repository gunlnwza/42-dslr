#!./venv/bin/python

import argparse
import sys

import pandas as pd

# NOTE: feel free to change anything: parsing, function signatures, ...


# TODO: implement with real logistic regression
def ft_predict(df: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame({
        "Index": df.index,
        "Hogwarts House": "Gryffindor"
    })
    return res


def main():
    parser = argparse.ArgumentParser(
        prog="logreg_predict.py",
        description="Predict which house the students belong to"
    )
    parser.add_argument("path", help=".csv file to predict")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.path, index_col="Index")
    except Exception as e:
        print(e)
        sys.exit(1)

    y_pred = ft_predict(df)
    y_pred.to_csv("houses.csv", index=False)


if __name__ == "__main__":
    main()
