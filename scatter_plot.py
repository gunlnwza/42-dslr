#!./venv/bin/python

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


def main():
    args = parse_path_feature_2()
    df = read_csv(args.path)
    scatter_plot(df, args.feature_x, args.feature_y)


if __name__ == "__main__":
    main()
