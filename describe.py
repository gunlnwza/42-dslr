#!./venv/bin/python

from utils import read_csv, parse_path 


# TODO: implement describe myself
def describe(df):
    print(df.describe())


def main():
    args = parse_path()
    df = read_csv(args.path)
    describe(df)


if __name__ == "__main__":
    main()
