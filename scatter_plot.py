#!./venv/bin/python

import matplotlib.pyplot as plt

from file_utils import read_csv, parse 


# TODO: implement
# TODO: code color by house
def scatter_plot(df):
    feature_x = "Arithmancy"
    feature_y = "Astronomy"
    plt.title(f"{feature_x} vs {feature_y}")
    plt.scatter(df[feature_x], df[feature_y])
    plt.show()    


def main():
    args = parse()
    df = read_csv(args.path)
    scatter_plot(df)


if __name__ == "__main__":
    main()
