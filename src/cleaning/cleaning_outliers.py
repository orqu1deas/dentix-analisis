# src / cleaning / cleaning_outliers.py

# Módulo de tratamiento de outliers para el proyecto Dentix.
# - Filtros de valores imposibles (reglas de negocio)
# - Winsorización por percentil (P99)
# - Transformaciones logarítmicas
# - Reportes detallados del impacto de cada transformación


import numpy as np
import pandas as pd


# 1. Filtros por reglas de negocio
def filter_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    df = df[df["PASIVOS"] < 1e12]

    removed = before - len(df)
    print(f"[Reglas de negocio] Registros eliminados por PASIVOS imposibles: {removed}")

    return df

# 2. Winsorización generalizada
def winsorize_p99(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    q99 = df[col].quantile(0.99)
    afectados = (df[col] > q99).sum()
    pct = afectados / len(df) * 100

    df[col] = df[col].clip(upper=q99)

    print(f"[Winsorización] {col}: {afectados} valores ({pct:.2f}%) — P99 = {q99:.2f}")

    return df

# 3. Transformación logarítmica segura
def add_log_transform(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()

    new_col = f"{col}_LOG"
    df[new_col] = np.log1p(df[col])

    print(f"[Transformación Log] {new_col} creada a partir de {col}")

    return df


# 4. Correcciones de variables específicas
def fix_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    invalidos = (df["SCORE"] > 999).sum()
    df.loc[df["SCORE"] > 999, "SCORE"] = 999

    print(f"[Corrección SCORE] {invalidos} valores ajustados a 999")

    return df


def fix_estrato(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ESTRATO"] = df["ESTRATO"].clip(1, 6)

    print("[Corrección ESTRATO] Rango ajustado a 1–6")

    return df


# 5. Pipeline completo de tratamiento de outliers
def clean_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = filter_impossible_values(df)

    columnas_winsor = ["CUOTAMENSUAL", "ACTIVOS", "PASIVOS"]
    for col in columnas_winsor:
        if col in df.columns:
            df = winsorize_p99(df, col)

    columnas_log = [
        "SALDO VENCIDO", "DIAS DE MORA ", "TOTAL INGRESOS",
        "INGRESOS FIJOS", "MONTO APROBADO", "MONTO PREAPROBADO",
        "PASIVOS"
    ]
    for col in columnas_log:
        if col in df.columns:
            df = add_log_transform(df, col)

    df = fix_score(df)
    df = fix_estrato(df)

    print("\n[Pipeline Outliers] Proceso completado.\n")

    return df
