#!./venv/bin/python

from utils import read_csv, parse_path 
from utils.stats.describe import display_stats, describe

def main():
    
    args = parse_path()
    
    df = read_csv(args.path)
    print(describe(df))

    # for rich table display
    # display_stats(df)


if __name__ == "__main__":
    main()
