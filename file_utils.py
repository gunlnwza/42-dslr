import argparse
import sys
import pandas as pd


def parse():
    parser = argparse.ArgumentParser(
        prog="describe.py",
        description="ft_describe",
    )
    parser.add_argument("path", help=".csv file to describe")
    args = parser.parse_args()
    return args


def read_csv(path: str):
    try:
        df = pd.read_csv(path, index_col="Index")
        return df
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
