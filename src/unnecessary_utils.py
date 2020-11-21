"""
This file contains random functions that were used for absolutely
random reasons without any explanation. There is always a reason,
but for now, just let it be.
"""
import pandas as pd

def decrease_biases_by_one(csv_path: str) -> None:
    """Decreases each bias by 10"""
    df = pd.read_csv(csv_path)
    df.loc[df['biases'], 'biases'] -= 10
    df.to_csv('../data/tweets_biases.csv')
