#!./venv/bin/python

from file_utils import read_csv, parse 


# TODO: implement
# TODO: code color by house
def pair_plot(df):
    pass


def main():
    args = parse()
    df = read_csv(args.path)
    pair_plot(df)


if __name__ == "__main__":
    main()
