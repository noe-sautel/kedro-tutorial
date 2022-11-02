"""
This is a boilerplate pipeline 'hello_modular'
generated using Kedro 0.18.3
"""

import pandas as pd

def get_mean(df: pd.DataFrame, col: str)->float:
    return df[col].mean()
