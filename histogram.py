#!./venv/bin/python

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.parse_utils import get_description


def plot_histogram(df: pd.DataFrame, col: str):
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(15, 10))
    plt.title(f"Distribution of '{col}' by Houses", fontsize=28, pad=20)
    colors = {"Gryffindor": "red", "Ravenclaw": "blue",
              "Hufflepuff": "gold", "Slytherin": "green"}
    sns.histplot(df, x=col, bins=20, hue="Hogwarts House", palette=colors)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="histogram.py",
        description=get_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help=".csv file to describe")
    parser.add_argument("feature", help="feature's name to plot")
    args = parser.parse_args()

    col = args.feature
    try:
        df = pd.read_csv(args.path, index_col="Index")
        if col not in df:
            raise ValueError(f"'{col}' is not in df")
    except Exception as e:
        print(e)
        sys.exit(1)

    plot_histogram(df, col)

 
if __name__ == "__main__":
    main()
