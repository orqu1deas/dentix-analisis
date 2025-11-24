# src / cleaning / cleaning_missing.py

# Manejo de aquellos valores faltantes para el proyecto Dentix.
# - Eliminación de columnas no útiles.
# - Filtrado mínimo por número de valores presentes.
# - Imputación tipo MICE (IterativeImputer) para variables financieras.
# - Regresión RandomForest para imputar TOTAL INGRESOS.
# - Conversión adecuada de columnas booleanas.
# - Reglas específicas para PERSONAS A CARGO.

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print(f"Filas originales: {len(df)}")

    # 1. Columnas a eliminar completamente
    cols_drop = ["DESISTIMIENTO", "FINANCIERA", "OTROS INGRESOS"]
    print((df.isnull().sum() / len(df) * 100).sort_values(ascending=False).head(25))

    df = df.drop(columns=cols_drop)
    print(f"Filas después del dropna: {len(df)}")

    # 2. Eliminar filas con demasiados NA (menos de 50 valores)
    df = df.dropna(thresh=50)
    print(f"Filas después del dropna: {len(df)}")

    # 3. Dropeo por filas según columnas importantes.
    cols_excluded = [
        "FINANCIERA", "DESISTIMIENTO", "OTROS INGRESOS", "TOTAL INGRESOS",
        "TOTAL EGRESOS", "OPERACION MONEDA EXTRAGERA", "NO PERSONAS A CARGO",
        "PASIVOS", "CUOTA DE CREDITOS", "ACTIVOS"
    ]

    cols_critical = [c for c in df.columns if c not in cols_excluded]

    df_clean = df.dropna(subset=cols_critical).copy()

    if "NO PERSONAS A CARGO" in df_clean.columns:
        df_clean["NO PERSONAS A CARGO"] = df_clean["NO PERSONAS A CARGO"].fillna(0)

    if "OPERACION MONEDA EXTRAGERA" in df_clean.columns:
        df_clean["OPERACION MONEDA EXTRAGERA"] = (
            df_clean["OPERACION MONEDA EXTRAGERA"]
            .astype(str)
            .map({"True": True, "False": False})
            .fillna(False)
        )

    mice_cols = ["TOTAL EGRESOS", "PASIVOS", "CUOTA DE CREDITOS", "ACTIVOS"]

    mice_imputer = IterativeImputer(random_state = 42, max_iter = 15)
    df_clean[mice_cols] = mice_imputer.fit_transform(df_clean[mice_cols])

    if "TOTAL INGRESOS" in df_clean.columns:
        mask_train = df_clean["TOTAL INGRESOS"].notnull()

        features = ["INGRESOS FIJOS", "CUOTA DE CREDITOS", "ACTIVOS",
                    "PASIVOS", "TOTAL EGRESOS"]
        
        X_train = df_clean.loc[mask_train, features]
        y_train = df_clean.loc[mask_train, "TOTAL INGRESOS"]

        rf = RandomForestRegressor(n_estimators = 100, max_depth = 8, random_state = 0)
        rf.fit(X_train, y_train)

        mask_pred = df_clean["TOTAL INGRESOS"].isnull()
        df_clean.loc[mask_pred, "TOTAL INGRESOS"] = rf.predict(
            df_clean.loc[mask_pred, features]
            )

    return df_clean
