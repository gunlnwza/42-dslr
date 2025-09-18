#!./venv/bin/python

from file_utils import read_csv, parse 


# TODO: implement describe myself
def describe(df):
    print(df.describe())


def main():
    args = parse()
    df = read_csv(args.path)
    describe(df)


if __name__ == "__main__":
    main()
