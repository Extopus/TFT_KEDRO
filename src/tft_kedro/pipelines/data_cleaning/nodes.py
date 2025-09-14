import pandas as pd

def clean_csv_data(df: pd.DataFrame) -> pd.DataFrame:
    # Eliminar filas con valores nulos
    df = df.dropna()
    # Eliminar duplicados
    df = df.drop_duplicates()
    # Opcional: resetear el índice
    df = df.reset_index(drop=True)
    return df
