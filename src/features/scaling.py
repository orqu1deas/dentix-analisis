# scaling.py

# Escalamiento de variables numéricas con StandardScaler para 
# estandarizar todas las columnas numéricas continuas, excepto
# aquellas que NO deben transformarse.



import pandas as pd
from sklearn.preprocessing import StandardScaler


def apply_scaling(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    cols_to_exclude = [
        "TRANSACTION ID",
        "IDENTIFICACION COMERCIAL",
        "CODIGO CONFIRMACION CREDITO",
        "PAGARE_ID",
        "NUMERO CREDITO TESEO",
        "NO CREDITO",
        "PLAZO",
        "SEGURO",
        "AVAL",
        "SEGURO_YJ",
        "AVAL_YJ"
    ]

    numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]

    scaler = StandardScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print(f"Escalado aplicado a {len(numeric_cols)} columnas numéricas:")
    print(numeric_cols)

    return df
