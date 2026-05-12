import pandas as pd
import os
import time
from datetime import datetime, timedelta
from scripts.tenis_api import safe_get
from dotenv import load_dotenv

load_dotenv(override=True)
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.api-tennis.com/tennis/?method=get_fixtures"

HIST_TYPES = {"Challenger Men Singles", "Challenger Men - Singles"}
FILE_PATH = "data/Challenger Tour Matches.csv"

def normalize_name(name):
    if not isinstance(name, str) or pd.isna(name):
        return ""
    # Quitar puntos (ej. T. Kumasaka -> t kumasaka)
    name = name.replace('.', '').strip().lower()
    return name

def fetch_fixtures_for_date(date_str):
    url = f"{BASE_URL}&APIkey={API_KEY}&date_start={date_str}&date_stop={date_str}"
    try:
        r = safe_get(url)
        if r.status_code == 200:
            data = r.json()
            if data.get("success") == 1:
                return data.get("result", [])
    except Exception as e:
        print(f"Error fetching {date_str}: {e}")
    return []

def recover_keys():
    print(f"Leyendo {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH)
    df['Fecha_dt'] = pd.to_datetime(df['Fecha_dt'])
    
    start_date = datetime(2026, 3, 24)
    end_date = datetime(2026, 4, 3)
    
    # Rango de fechas a procesar
    delta = end_date - start_date
    dates_to_process = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(delta.days + 1)]
    
    updates_total = 0
    
    for date_str in dates_to_process:
        print(f"Procesando fecha: {date_str}...")
        api_results = fetch_fixtures_for_date(date_str)
        
        # Filtrar solo Challenger Singles
        challenger_fixtures = [f for f in api_results if f.get("event_type_type") in HIST_TYPES]
        
        if not challenger_fixtures:
            print(f"  No se encontraron fixtures de Challenger para esta fecha.")
            continue
            
        # Filtrar DataFrame para esta fecha y sin ID
        mask_csv = (df['Fecha_dt'] == date_str) & (df['ID Partido'].isna() | (df['ID Partido'].astype(str).str.strip() == ''))
        rows_to_update = df[mask_csv]
        
        if rows_to_update.empty:
            print(f"  No hay partidos pendientes en el CSV para esta fecha.")
            continue
            
        print(f"  Intentando emparejar {len(rows_to_update)} partidos...")
        
        date_updates = 0
        for idx, row in rows_to_update.iterrows():
            p1_csv = normalize_name(row['Jugador 1'])
            p2_csv = normalize_name(row['Jugador 2'])
            set_csv = {p1_csv, p2_csv}
            
            for f in challenger_fixtures:
                p1_api = normalize_name(f.get("event_first_player"))
                p2_api = normalize_name(f.get("event_second_player"))
                set_api = {p1_api, p2_api}
                
                if set_csv == set_api:
                    event_key = f.get("event_key")
                    df.at[idx, 'ID Partido'] = event_key
                    date_updates += 1
                    break
        
        print(f"  ÉXITO: Se actualizaron {date_updates} IDs.")
        updates_total += date_updates
        time.sleep(0.5) # Respetar rate limits
        
    if updates_total > 0:
        # Guardar cambios
        # Asegurar que ID Partido sea string para no perder ceros o convertir a float
        df['ID Partido'] = df['ID Partido'].fillna('').astype(str).replace('\.0', '', regex=True)
        df.to_csv(FILE_PATH, index=False)
        print(f"\nProceso finalizado. Se actualizaron {updates_total} partidos en total.")
    else:
        print("\nNo se encontraron coincidencias para actualizar.")

if __name__ == "__main__":
    recover_keys()
