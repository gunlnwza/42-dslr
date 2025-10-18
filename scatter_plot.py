#!./venv/bin/python

import argparse

import matplotlib.pyplot as plt

from utils import read_csv, parse_path_feature_2


# TODO: iterate until perfect
def scatter_plot(df, feature_x: str, feature_y: str):
    plt.title(f"{feature_x} VS {feature_y}")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)

    gryffindor = df[df["Hogwarts House"] == "Gryffindor"]
    ravenclaw = df[df["Hogwarts House"] == "Ravenclaw"]
    hufflepuff = df[df["Hogwarts House"] == "Hufflepuff"]
    slytherin = df[df["Hogwarts House"] == "Slytherin"]

    plt.scatter(gryffindor[feature_x], gryffindor[feature_y], color="red", alpha=0.4, label="Gryffindor")
    plt.scatter(ravenclaw[feature_x], ravenclaw[feature_y], color="blue", alpha=0.4, label="Ravenclaw")
    plt.scatter(hufflepuff[feature_x], hufflepuff[feature_y], color="yellow", alpha=0.4, label = "Hufflepuff")
    plt.scatter(slytherin[feature_x], slytherin[feature_y], color="green", alpha=0.4, label="Slytherin")

    plt.legend()
    plt.show()


def get_description():
    COLUMN_NAMES = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination",
        "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions",
        "Care of Magical Creatures", "Charms", "Flying"
    ]
    
    des = "Draw a scatter plot for the selected columns:\n"
    for c in COLUMN_NAMES:
        des += f"  {c}\n"
    return des


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

    df = read_csv(args.path)
    scatter_plot(df, args.feature_x, args.feature_y)


if __name__ == "__main__":
    main()
