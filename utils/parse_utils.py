import argparse


def parse_path():
    parser = argparse.ArgumentParser(
        prog="describe.py",
        description="Describe numerical columns' statistics",
    )
    parser.add_argument("path", help=".csv file to describe")
    args = parser.parse_args()
    return args


def parse_path_feature():
    parser = argparse.ArgumentParser(
        prog="histogram.py",
        description="Plot Histogram",
        add_help=False
    )
    parser.add_argument("-h", action="store_true", help="Help message")
    parser.add_argument("path", help=".csv file to describe")
    parser.add_argument("feature", help="Feature name")
    args = parser.parse_args()
    return args


def parse_path_feature_2():
    parser = argparse.ArgumentParser(
        prog="histogram.py",
        description="Plot Histogram",
        add_help=False
    )
    parser.add_argument("-h", action="store_true", help="Help message")
    parser.add_argument("path", help=".csv file to describe")
    parser.add_argument("feature_x", help="Feature X name")
    parser.add_argument("feature_y", help="Feature Y name")
    args = parser.parse_args()
    return args
