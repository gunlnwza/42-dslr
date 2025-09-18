#!./venv/bin/python

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


def main():
    args = parse_path_feature()
    if args.h:
        print("Arithmancy\nAstronomy\nHerbology\nDefense Against the Dark Arts\nDivination\nMuggle Studies\nAncient Runes\nHistory of Magic\nTransfiguration\nPotions\nCare of Magical Creatures\nCharms\nFlying")
    df = read_csv(args.path)
    histogram(df, args.feature)


if __name__ == "__main__":
    main()
