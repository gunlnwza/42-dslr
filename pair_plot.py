#!./venv/bin/python

import argparse
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.parse_utils import get_description

# TODO: do csv validation

# TODO: improve plotting ux, make auto complete (argcomplete)
# import argcomplete


def pair_plot(df: pd.DataFrame, cols: list[str]):
    plt.rcParams.update({'font.size': 12})

    colors = {"Gryffindor": "red", "Ravenclaw": "blue",
              "Hufflepuff": "gold", "Slytherin": "green"}
    sns.pairplot(df[cols + ["Hogwarts House"]], hue="Hogwarts House", palette=colors)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="scatter_plot.py",
        description=get_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help=".csv file to describe")
    parser.add_argument("features", help="features' name to plot", nargs="+")
    args = parser.parse_args()

    # TODO: since the plot can only fit 8 features, check that it is <= 8 features, checking >= 2 would also be nice

    cols: list[str] = args.features
    try:
        df = pd.read_csv(args.path, index_col="Index")
        for col in cols:
            if col not in df:
                raise ValueError(f"'{col}' is not in df")
    except Exception as e:
        print(e)
        sys.exit(1)

    pair_plot(df, cols)


if __name__ == "__main__":
    main()
