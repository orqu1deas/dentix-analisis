import pandas as pd
from data_loading import load_base
from cleaning.cleaning_basic import basic_cleaning
from cleaning.commercial_fix import fix_commercial
from cleaning.cleaning_missing import clean_missing_values
from cleaning.cleaning_outliers import clean_outliers

def full_clean_pipeline(save_intermediate: bool = True) -> pd.DataFrame:
    df = load_base("data/raw/BaseDentix.xlsx")
    
    df = basic_cleaning(df)
    if save_intermediate:
        df.to_csv("data/interim/01_basic_cleaning.csv", index=False)

    df = fix_commercial(df)
    if save_intermediate:
        df.to_csv("data/interim/02_commercial_fixed.csv", index=False)
    
    df = clean_missing_values(df)
    if save_intermediate:
        df.to_csv("data/interim/03_missing_imputed.csv", index=False)
    
    df = clean_outliers(df)
    if save_intermediate:
        df.to_csv("data/interim/04_outliers_cleaned.csv", index=False)
    return df

def main():
    print("Ejecutando pipeline de limpieza Dentix...")
    df_clean = full_clean_pipeline()

    print("\nPipeline completado.")
    print(f"Registros finales: {len(df_clean)}")

    output_path = "data/processed/dentix_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"Datos limpios guardados en: {output_path}")

if __name__ == "__main__":
    main()
