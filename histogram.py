#!./venv/bin/python

import matplotlib.pyplot as plt

from file_utils import read_csv, parse 


# TODO: implement
# TODO: code color by house
def histogram(df):
    feature = "Arithmancy"
    plt.title(f"{feature} Score Distribution")
    plt.hist(df[feature])
    plt.show()    


def main():
    args = parse()
    df = read_csv(args.path)
    histogram(df)


if __name__ == "__main__":
    main()
