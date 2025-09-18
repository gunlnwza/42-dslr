#!./venv/bin/python

import matplotlib.pyplot as plt

from utils import read_csv, parse_path_feature_2


# TODO: implement, use histogram() and scatter_plot()
def pair_plot():
    pass


def main():
    args = parse_path_feature_2()
    df = read_csv(args.path)
    pair_plot(df, args.feature_x, args.feature_y)


if __name__ == "__main__":
    main()
