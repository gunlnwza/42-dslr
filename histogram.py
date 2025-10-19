#!./venv/bin/python

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMN_NAMES = [
    "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination",
    "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions",
    "Care of Magical Creatures", "Charms", "Flying"
]


def get_description():
    des = "Draw a histogram for the selected column:\n"
    for c in COLUMN_NAMES:
        des += f"  {c}\n"
    return des


def plot_histogram(df: pd.DataFrame, col: str, separate: bool):
    sns.set_style("darkgrid")
    plt.rcParams.update({'font.size': 20})

    plt.figure(figsize=(15, 10))
    plt.xlabel(col)
    plt.ylabel("Count")
    if separate:
        plt.title(f"Distribution of '{col}' by House")
        colors = {
            "Gryffindor": "red",
            "Ravenclaw": "blue",
            "Hufflepuff": "gold",
            "Slytherin": "green",
        }
        for house, group in df.groupby("Hogwarts House"):
            plt.hist(group[col], bins=20, alpha=0.6, label=house, color=colors.get(house, "gray"))
        plt.legend()
    else:
        plt.title(f"Distribution of '{col}'", fontsize=28)
        plt.hist(df[col], alpha=0.8)
    
    plt.show()


def main():    
    parser = argparse.ArgumentParser(
        prog="histogram.py",
        description=get_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-s", "--separate", action="store_true", help="Divide the dataset by houses")
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

    plot_histogram(df, col, args.separate)

 
if __name__ == "__main__":
    main()
