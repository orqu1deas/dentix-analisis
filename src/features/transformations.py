
# src / features / transformations.py

# Transformacion Yeo–Johnson del proyecto Dentix que forman
# parte del pipeline de preparación de datos.


import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer


def apply_yeo_johnson(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    yeo_vars = [
        "INGRESOS FIJOS",
        "TOTAL INGRESOS",
        "GASTOS DE SOSTENIMIENTO",
        "MONTO PREAPROBADO",
        "MONTO APROBADO",
        "MONTO DESEMBOLSO",
        "SEGURO",
        "AVAL",
        "EDAD"
    ]

    yeo_vars = [col for col in yeo_vars if col in df.columns]

    if not yeo_vars:
        print("[Yeo–Johnson] No se encontraron columnas aplicables.")
        return df

    pt = PowerTransformer(method="yeo-johnson")

    transformed = pt.fit_transform(df[yeo_vars])

    new_cols = [col + "_YJ" for col in yeo_vars]
    df[new_cols] = transformed

    print(f"[Yeo–Johnson] Transformadas: {', '.join(yeo_vars)}")
    return df
