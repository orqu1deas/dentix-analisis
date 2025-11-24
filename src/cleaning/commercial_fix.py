# src / cleaning / commercial_fix.py

# Corrige valores inconsistentes en la columna COMERCIAL.

import pandas as pd
import re


def prefix_ok(comercial: str) -> bool:
    """Valida si el comercial cumple la estructura esperada."""
    return bool(re.match(r"^(colte_|dentix_)", comercial))


def fix_commercial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia valores inconsistentes en la columna COMERCIAL
    usando reglas basadas en IDENTIFICACION COMERCIAL.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame con COMERCIAL corregido.
    """

    df = df.copy()

    comercial_count = df.groupby("IDENTIFICACION COMERCIAL")["COMERCIAL"].nunique()
    ids_multiples = comercial_count[comercial_count > 1].index

    tuplas_validas = (
        df[df["IDENTIFICACION COMERCIAL"].isin(ids_multiples)]
        .groupby(["IDENTIFICACION COMERCIAL", "COMERCIAL"])
        .size()
        .index.tolist()
    )

    for comercial in df["COMERCIAL"].unique():
        if not prefix_ok(comercial):

            for id_com, com_valido in tuplas_validas:
                if comercial == com_valido:

                    candidatos = [
                        c for (id_c, c) in tuplas_validas
                        if id_c == id_com and prefix_ok(c)
                    ]

                    if candidatos:
                        df.loc[df["COMERCIAL"] == comercial, "COMERCIAL"] = candidatos[0]

                    break

    return df
