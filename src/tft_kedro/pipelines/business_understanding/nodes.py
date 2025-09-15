import pandas as pd

def describe_dataset(df: pd.DataFrame) -> dict:
    """Genera un resumen estadístico del DataFrame."""
    return {
        "head": df.head().to_dict(),
        "info": str(df.info()),
        "describe": df.describe().to_dict(),
        "shape": df.shape
    }
