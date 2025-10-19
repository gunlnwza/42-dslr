#!./venv/bin/python

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.parse_utils import get_description


def scatter_plot(df, col_x: str, col_y: str, separate: bool):
    sns.set_style("darkgrid")
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(15, 10))
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    if separate:
        plt.title(f"Scatter Plot of '{col_x}' vs '{col_y}' by Houses", fontsize=28)
        colors = {
            "Gryffindor": "red",
            "Ravenclaw": "blue",
            "Hufflepuff": "gold",
            "Slytherin": "green",
        }
        for house, group in df.groupby("Hogwarts House"):
            plt.scatter(group[col_x], group[col_y], alpha=0.7, label=house, color=colors.get(house, "gray"))
        plt.legend()
    else:
        plt.title(f"Scatter Plot of '{col_x}' vs '{col_y}'", fontsize=28)
        plt.scatter(df[col_x], df[col_y], alpha=0.8)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="scatter_plot.py",
        description=get_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-s", "--separate", action="store_true", help="Divide the dataset by houses")
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

    scatter_plot(df, col_x, col_y, args.separate)


if __name__ == "__main__":
    main()
