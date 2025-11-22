# src / cleaning / cleaning_basic.py

# Limpieza inicial de la base Dentix:
# - Eliminación de duplicados
# - Corrección simple de valores inválidos en SCORE y PERSONAS A CARGO

import numpy as np
import pandas as pd


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # 1. Eliminar duplicados
    df = df.drop_duplicates()

    # 2. Reglas de negocio – SCORE debe estar entre 0 y 1000
    invalid_score = (df["SCORE"] < 0) | (df["SCORE"] > 1000)
    df.loc[invalid_score, "SCORE"] = np.nan

    # 3. Reglas de negocio – No. personas a cargo debe ser razonable
    invalid_dependents = (df["NO PERSONAS A CARGO"] < 0) | (df["NO PERSONAS A CARGO"] > 50)
    df.loc[invalid_dependents, "NO PERSONAS A CARGO"] = np.nan

    return df
