# src / data_loading.py
# Módulo encargado de la carga de datos del proyecto Dentix.

import pandas as pd


def load_base(path: str = "data/BaseDentix.xlsx") -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
        return df
    
    except Exception as e:
        raise RuntimeError(f"Error al cargar la base desde {path}: {e}")
