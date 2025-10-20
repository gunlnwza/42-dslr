#!./venv/bin/python

import argparse
import sys

import numpy as np
import pandas as pd

from utils.stats.describe import ft_describe, display_stats


def main():
    parser = argparse.ArgumentParser("describe.py", description="describe of statistics of numerical columns")
    parser.add_argument("-p", "--pretty", action="store_true", help="pretty print table")
    parser.add_argument("path", help=".csv file to describe")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.path, index_col="Index")
    except Exception as e:
        print(e)
        sys.exit(1)

    if args.pretty:
        display_stats(df)  # rich table display
    else:
        pd.options.display.float_format = '{:.6f}'.format
        print(ft_describe(df))


if __name__ == "__main__":
    main()
