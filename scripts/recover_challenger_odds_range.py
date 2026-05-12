import pandas as pd
from scripts.tenis_api import get_odds_data
import os
import time
from datetime import datetime, timedelta

# Configuración
FILE_PATH = "data/Challenger Tour Matches.csv"

def recover_odds_range():
    print("Iniciando recuperación de cuotas Challenger para el rango 24 Mar - 3 Abr...")
    df = pd.read_csv(FILE_PATH)
    df['Fecha_dt'] = pd.to_datetime(df['Fecha_dt'])
    
    start_date = datetime(2026, 3, 24)
    end_date = datetime(2026, 4, 3)
    
    # Identificar partidos en el rango con ID pero sin cuotas
    mask = (df['Fecha_dt'] >= start_date) & (df['Fecha_dt'] <= end_date) & \
           (df['ID Partido'].notna()) & (df['ID Partido'].astype(str) != '') & \
           ((df['Cuota J1'] == 0) | (df['Cuota J1'].isna()))
    
    to_recover = df[mask].copy()
    
    if to_recover.empty:
        print("No hay partidos pendientes de recuperación de cuotas en este rango.")
        return

    print(f"Encontrados {len(to_recover)} partidos para recuperar cuotas.")
    
    # Agrupar por fecha para optimizar llamadas a la API
    dates = sorted(to_recover['Fecha_dt'].unique())
    
    odds_cache = {}
    for date_ts in dates:
        date_str = date_ts.strftime("%Y-%m-%d")
        print(f"Buscando cuotas para la fecha: {date_str}...")
        # get_odds_data espera YYYY-MM-DD
        day_odds = get_odds_data(date_str, date_str)
        if day_odds:
            print(f"  Encontradas cuotas para {len(day_odds)} partidos.")
            odds_cache.update(day_odds)
        else:
            print(f"  No se encontraron cuotas para esta fecha.")
        time.sleep(0.5) # Respetar rate limits
    
    # Actualizar el DataFrame original
    updated_count = 0
    for idx, row in to_recover.iterrows():
        match_id = str(row['ID Partido']).replace('.0', '').strip()
        if match_id in odds_cache:
            o1, o2 = odds_cache[match_id]
            df.at[idx, 'Cuota J1'] = o1
            df.at[idx, 'Cuota J2'] = o2
            updated_count += 1
            
    if updated_count > 0:
        # Asegurar que ID Partido se mantenga bien formateado antes de guardar
        df['ID Partido'] = df['ID Partido'].fillna('').astype(str).replace('\.0', '', regex=True)
        df.to_csv(FILE_PATH, index=False)
        print(f"\nProceso finalizado. Se actualizaron {updated_count} partidos con nuevas cuotas.")
    else:
        print("\nNo se encontraron cuotas disponibles en el periodo consultado.")

if __name__ == "__main__":
    recover_odds_range()
