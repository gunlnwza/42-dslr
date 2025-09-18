import sys
import pandas as pd


def read_csv(path: str):
    try:
        df = pd.read_csv(path, index_col="Index")
        return df
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
