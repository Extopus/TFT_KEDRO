import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gameDuration_min'] = df['gameDuration'] / 60
    # Agrega aquí más transformaciones si lo deseas
    return df
