# src / features / encoding.py

# Codificación de variables categóricas para el proyecto Dentix.

import pandas as pd
import numpy as np

# 1. Ordinal Encoding — Nivel de estudios
def encode_nivel_estudios(df: pd.DataFrame) -> pd.DataFrame:
    orden_niveles = {
        'Primaria': 0,
        'Bachillerato': 1,
        'Técnico': 2,
        'Tecnólogo': 3,
        'Licenciatura': 4,
        'Universitario': 5,
        'Especialización': 6,
        'Maestría': 7,
        'Doctorado / Postdoctorado': 8
    }

    df["NIVEL ESTUDIOS_ORD"] = df["NIVEL ESTUDIOS"].map(orden_niveles)
    return df

# 2. Frequency Encoding — alta cardinalidad
def frequency_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    freqs = df[col].value_counts(normalize=True)
    df[col + "_FREQ"] = df[col].map(freqs)
    return df


# 3. Rare Encoding — cardinalidad extrema
def rare_encoding(df: pd.DataFrame, col: str, threshold: float = 0.01) -> pd.DataFrame:
    freqs = df[col].value_counts(normalize=True)
    rare_labels = freqs[freqs < threshold].index
    df[col] = df[col].replace(rare_labels, "OTRO")
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    onehot_cols = [
        "ESTADOCIVIL",
        "TIPO VIVIENDA",
        "ACTIVIDAD ECONÓMICA",
        "TIPO CONTRATO",
        "OCUPACIÓN",
        "LUGAR NACIMIENTO",
        "INCIDENCIAFORMALIZACION",
        "CLINICA"
    ]

    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    return df


def apply_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = encode_nivel_estudios(df)

    rare_cols = ["DIRECCION", "BARRIO", "EMPRESA"]
    for col in rare_cols:
        if col in df.columns:
            df = rare_encoding(df, col, threshold=0.01)

    freq_cols = ["PROFESION", "CIUDAD", "CIUDAD_LIMPIA", "COMERCIAL", "DEPARTAMENTO"]
    for col in freq_cols:
        if col in df.columns:
            df = frequency_encoding(df, col)

    # D) One-Hot Encoding
    df = one_hot_encode(df)

    return df
