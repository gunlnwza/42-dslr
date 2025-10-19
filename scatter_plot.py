#!./venv/bin/python

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.parse_utils import get_description


def scatter_plot(df, col_x: str, col_y: str):
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(15, 10))
    plt.title(f"Scatter Plot of '{col_x}' vs '{col_y}' by Houses", fontsize=28, pad=20)
    colors = {"Gryffindor": "red", "Ravenclaw": "blue",
              "Hufflepuff": "gold", "Slytherin": "green"}
    sns.scatterplot(df, x=col_x, y=col_y, hue="Hogwarts House", palette=colors, s=100)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="scatter_plot.py",
        description=get_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help=".csv file to describe")
    parser.add_argument("feature_x", help="feature's name to plot on x-axis")
    parser.add_argument("feature_y", help="feature's name to plot on y-axis")
    args = parser.parse_args()

    col_x = args.feature_x
    col_y = args.feature_y
    try:
        df = pd.read_csv(args.path, index_col="Index")
        if col_x not in df:
            raise ValueError(f"'{col_x}' is not in df")
        if col_y not in df:
            raise ValueError(f"'{col_y}' is not in df")
    except Exception as e:
        print(e)
        sys.exit(1)

    scatter_plot(df, col_x, col_y)


if __name__ == "__main__":
    main()
