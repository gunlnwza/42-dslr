import numpy as np
import pandas as pd

from rich.text import Text
from rich.table import Table
from rich.console import Console

from utils.stats import aggregation as agg

def ft_describe(df: pd.DataFrame) -> pd.DataFrame:
    """mimic pandas' describe, with extra twists"""

    numerical_cols = df.select_dtypes(include=np.number).columns

    statistics = ['count', 'mean', 'stddev', 'variance', 'min', '25%', '50%', '75%', 'IQR', 'max', 'skewness', 'kurtosis']
    results = {stat: {} for stat in statistics}

    for col_name in numerical_cols:

        column = df[col_name].to_numpy()
        
        results['count'][col_name] = agg.ft_count(column)
        results['mean'][col_name] = agg.ft_mean(column)
        results['stddev'][col_name] = agg.ft_stddev(column)
        results['min'][col_name] = agg.ft_min(column)
        results['max'][col_name] = agg.ft_max(column)

        results['25%'][col_name] = agg.ft_percentile(column, 25)
        results['50%'][col_name] = agg.ft_percentile(column, 50)
        results['75%'][col_name] = agg.ft_percentile(column, 75)

        results['variance'][col_name] = agg.ft_variance(column)
        results['IQR'][col_name] = agg.ft_iqr(column)
        results['skewness'][col_name] = agg.ft_skewness(column)
        results['kurtosis'][col_name] = agg.ft_kurtosis(column)
        

    stats_df = pd.DataFrame(results).T
    stats_df = stats_df.reindex(statistics)

    return stats_df


def display_stats(df: pd.DataFrame) -> None:

    console = Console()
    stats_df = ft_describe(df)
    
    table = Table(title="Dataframe Statistics", show_lines=True, header_style="bold magenta")
    table.add_column("Statistic", style="bold cyan", justify="left")

    for col in stats_df.columns:
        table.add_column(Text(col, style="italic green"), justify="right")

    for index, row in stats_df.iterrows():
        row_data = [index]
        
        for value in row:

            if pd.isna(value):
                row_data.append(Text("NaN", style="dim red"))

            elif index == 'count':
                row_data.append(f"{int(value):,}")

            else:
                row_data.append(f"{value:,.6f}")
        
        table.add_row(*row_data)

    console.print(table)
