#!./venv/bin/python

import argparse
import sys

import matplotlib.pyplot as plt

from utils import read_csv, parse_path_feature 


# TODO: iterate until perfect
def histogram(df, feature: str):
    plt.title(f"{feature} Score Distribution")

    gryffindor = df[df["Hogwarts House"] == "Gryffindor"]
    ravenclaw = df[df["Hogwarts House"] == "Ravenclaw"]
    hufflepuff = df[df["Hogwarts House"] == "Hufflepuff"]
    slytherin = df[df["Hogwarts House"] == "Slytherin"]

    plt.hist(gryffindor[feature], color="red", alpha=0.4, label="Gryffindor")
    plt.hist(ravenclaw[feature], color="blue", alpha=0.4, label="Ravenclaw")
    plt.hist(hufflepuff[feature], color="yellow", alpha=0.4, label = "Hufflepuff")
    plt.hist(slytherin[feature], color="green", alpha=0.4, label="Slytherin")

    plt.legend()
    plt.show()


def get_description():
    COLUMN_NAMES = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination",
        "Muggle Studies", "Ancient Runes", "History of Magic", "Transfiguration", "Potions",
        "Care of Magical Creatures", "Charms", "Flying"
    ]
    
    des = "Draw a histogram for the selected column:\n"
    for c in COLUMN_NAMES:
        des += f"  {c}\n"
    return des


def main():    
    parser = argparse.ArgumentParser(
        prog="histogram.py",
        description=get_description(),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("path", help=".csv file to describe")
    parser.add_argument("feature", help="feature's name to plot")
    args = parser.parse_args()

    df = read_csv(args.path)
    histogram(df, args.feature)


if __name__ == "__main__":
    main()
