# src / prepare_dataset.py

# Genera el dataset FINAL listo para modelado del proyecto Dentix.

import pandas as pd

from features.transformations import apply_yeo_johnson
from features.encoding import apply_encoding
from features.scaling import apply_scaling

COLS_TO_DROP = [
    # Fechas
    "FECHA NACIMIENTO", "FECHA EXP DOC", "FECHA SOLICITUD",
    "FECHA APROBACIÓN", "FECHA DESEMBOLSO", "FCREACION DENTICUOTAS",

    # Categóricas originales (ya codificadas)
    "NIVEL ESTUDIOS", "PROFESION", "GÉNERO", "CIUDAD", "DEPARTAMENTO",
    "DIRECCION", "BARRIO", "EMPRESA", "TIEMPO ACTIVIDAD", "COMERCIAL",
    "CLINICA", "LUGAR NACIMIENTO", "INCIDENCIAFORMALIZACION", "ESTADOCIVIL",
    "TIPO VIVIENDA", "ACTIVIDAD ECONÓMICA", "TIPO CONTRATO", "OCUPACIÓN",

    # "FRANJA DE MORA",
]

def prepare_final_dataset(df: pd.DataFrame, save_path: str = "data/features/dentix_model_input.csv") -> pd.DataFrame:
    df = df.copy()

    df = apply_yeo_johnson(df)

    df = apply_encoding(df)

    cols_to_drop_final = [c for c in COLS_TO_DROP if c in df.columns]
    df.drop(columns=cols_to_drop_final, inplace=True)

    df = apply_scaling(df)

    df.to_csv(save_path, index=False)

    print(f"Dataset listo guardado en: {save_path}")
    print(f"Total columnas finales: {len(df.columns)}")

    return df

def main():
    print("\nEjecutando prepare_final_dataset.py...")

    input_path = "data/processed/dentix_clean.csv"

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No se encontró el archivo {input_path}. "
            "Asegúrate de haber corrido antes el pipeline de limpieza."
        )
    
    prepare_final_dataset(df)

    print("Proceso completado.\n")


if __name__ == "__main__":
    main()