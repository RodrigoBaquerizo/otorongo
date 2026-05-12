import pandas as pd
from scripts.tenis_api import get_odds_data
import logging
from datetime import datetime

# Configuración
FILE_PATH = "data/Challenger Tour Matches.csv"

def recover_odds():
    print("Iniciando recuperación de cuotas Challenger...")
    df = pd.read_csv(FILE_PATH)
    
    # Identificar partidos con ID pero sin cuotas
    mask = df['ID Partido'].notna() & (df['ID Partido'].astype(str) != '') & (df['Cuota J1'].isna() | (df['Cuota J1'] == 0))
    to_recover = df[mask].copy()
    
    if to_recover.empty:
        print("No hay partidos pendientes de recuperación con ID Partido.")
        return

    print(f"Encontrados {len(to_recover)} partidos para recuperar.")
    
    # Agrupar por fecha para optimizar llamadas a la API
    dates = to_recover['Fecha_dt'].unique()
    
    odds_cache = {}
    for date_str in dates:
        print(f"Buscando cuotas para la fecha: {date_str}...")
        # get_odds_data espera YYYY-MM-DD
        day_odds = get_odds_data(date_str, date_str)
        odds_cache.update(day_odds)
    
    # Actualizar el DataFrame original
    updated_count = 0
    for idx, row in to_recover.iterrows():
        match_id = str(row['ID Partido']).replace('.0', '')
        if match_id in odds_cache:
            o1, o2 = odds_cache[match_id]
            df.at[idx, 'Cuota J1'] = o1
            df.at[idx, 'Cuota J2'] = o2
            updated_count += 1
            
    if updated_count > 0:
        df.to_csv(FILE_PATH, index=False)
        print(f"Proceso finalizado. Se actualizaron {updated_count} partidos.")
    else:
        print("No se encontraron cuotas en la API para los partidos identificados.")

if __name__ == "__main__":
    recover_odds()
