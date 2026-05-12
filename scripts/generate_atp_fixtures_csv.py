import pandas as pd
import requests
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

import time

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar variables de entorno
load_dotenv(override=True)
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.api-tennis.com/tennis/?method=get_fixtures"

def get_date_ranges(start_date_str, end_date_str, delta_days=7):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    ranges = []
    
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=delta_days), end_date)
        ranges.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end + timedelta(days=1)
    return ranges

def fetch_fixtures_chunk(from_date, to_date, retries=3):
    url = f"{BASE_URL}&APIkey={API_KEY}&date_start={from_date}&date_stop={to_date}"
    for i in range(retries):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") == 1:
                    return data.get("result", [])
            elif response.status_code == 500:
                logging.warning(f"  Intento {i+1} fallido (500) para {from_date} a {to_date}. Reintentando...")
            else:
                logging.error(f"  Error HTTP {response.status_code} para {from_date} a {to_date}")
        except Exception as e:
            logging.error(f"  Error en intento {i+1}: {e}")
        
        if i < retries - 1:
            time.sleep(2 ** (i + 1)) # Backoff exponencial
            
    return None # Significa que fallaron todos los reintentos

def main():
    start_date = "2024-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    output_file = "data/atp_challenger_fixtures_2024_2026.csv"
    tournaments_file = "data/tournaments.csv"
    
    # 1. Cargar mapeo de superficies
    if not os.path.exists(tournaments_file):
        logging.error(f"Archivo {tournaments_file} no encontrado.")
        return
    
    df_trn = pd.read_csv(tournaments_file, usecols=["tournament_key", "tournament_sourface"])
    surface_map = df_trn.set_index("tournament_key")["tournament_sourface"].to_dict()
    
    # 2. Generar rangos de fechas (7 días para ser más granulares y evitar errores 500)
    date_ranges = get_date_ranges(start_date, end_date, delta_days=7)
    logging.info(f"Generando datos en {len(date_ranges)} bloques semanales...")
    
    all_fixtures = []
    failed_ranges = []
    
    # Tipos de partidos a incluir (incluyendo variantes detectadas en la API)
    INCLUDED_TYPES = {"Atp Singles", "Challenger Men Singles", "Challenger Men - Singles"}
    
    # 3. Descargar datos
    for from_date, to_date in date_ranges:
        logging.info(f"Descargando bloque: {from_date} a {to_date}...")
        results = fetch_fixtures_chunk(from_date, to_date)
        
        if results is not None:
            filtered = [r for r in results if r.get("event_type_type") in INCLUDED_TYPES]
            all_fixtures.extend(filtered)
            atp_count = sum(1 for r in filtered if r.get("event_type_type") == "Atp Singles")
            ch_count = sum(1 for r in filtered if r.get("event_type_type") == "Challenger Men Singles")
            logging.info(f"  OK. ATP Singles: {atp_count}, Challenger: {ch_count}")
        else:
            logging.error(f"  CRÍTICO: Fallaron todos los reintentos para {from_date} a {to_date}. Saltando...")
            failed_ranges.append((from_date, to_date))
        
        time.sleep(0.5) # Pausa mínima para no saturar
            
    if not all_fixtures:
        logging.error("No se encontraron partidos ATP Singles.")
        return
        
    logging.info(f"Total de partidos encontrados: {len(all_fixtures)}. Guardando CSV...")
    if failed_ranges:
        logging.warning(f"Se saltaron {len(failed_ranges)} rangos por errores persistentes.")

    # 4. Procesar y Guardar
    processed_data = []
    for f in all_fixtures:
        winner = "-"
        if f.get("event_winner") == "First Player":
            winner = f.get("event_first_player")
        elif f.get("event_winner") == "Second Player":
            winner = f.get("event_second_player")
        
        t_key = f.get("tournament_key")
        try:
            t_key_int = int(t_key) if t_key and str(t_key).isdigit() else t_key
            surface = surface_map.get(t_key_int, "Unknown")
        except:
            surface = "Unknown"
        
        processed_data.append({
            "Fecha": f.get("event_date"),
            "Torneo": f.get("tournament_name"),
            "Superficie": surface,
            "Jugador 1": f.get("event_first_player"),
            "Jugador 2": f.get("event_second_player"),
            "Ganador": winner
        })
    
    df_final = pd.DataFrame(processed_data)
    
    # Normalizar superficies (igual que process_files.py)
    import numpy as np
    
    # 1. Normalización general
    conditions = [
        df_final["Superficie"].astype(str).str.contains("Hard", case=False, na=False),
        df_final["Superficie"].astype(str).str.contains("Clay", case=False, na=False),
        df_final["Superficie"].astype(str).str.contains("Grass", case=False, na=False),
    ]
    df_final["Superficie"] = np.select(conditions, ["Hard", "Clay", "Grass"], default=df_final["Superficie"])
    
    # 2. Normalización específica (Davis Cup y casos Unknown conocidos)
    df_final.loc[df_final["Torneo"].str.contains("Davis", case=False, na=False), "Superficie"] = "Hard"
    df_final.loc[df_final["Torneo"] == "Brasilia 2", "Superficie"] = "Clay"
    df_final.loc[df_final["Torneo"] == "Fujairah", "Superficie"] = "Hard"
    df_final.loc[df_final["Torneo"] == "Metepec", "Superficie"] = "Hard"
    
    df_final.sort_values(by="Fecha", ascending=False).to_csv(output_file, index=False, encoding="utf-8")
    logging.info(f"Archivo generado: {output_file} ({len(processed_data)} registros)")


if __name__ == "__main__":
    main()
