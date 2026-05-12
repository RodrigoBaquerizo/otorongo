import pandas as pd
import os
import sys
import json
import logging
from datetime import datetime

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Añadir el directorio raíz al path para importar refresh_data
sys.path.append(os.getcwd())
from scripts.refresh_data import (
    calc_performance_refresh, 
    calc_ultra_performance_refresh, 
    load_player_master
)

def fix_all_performance_universal():
    """
    Saneamiento Universal 2026: Recalcula TODAS las filas de rendimiento en el CSV de 2026
    basándose en el historial completo (ATP + Challenger).
    """
    atp_file = "data/ATP Tour 2026 Matches.csv"
    hist_file = "data/atp_challenger_fixtures_2024_2026.csv"
    
    if not os.path.exists(atp_file):
        logging.error(f"No se encuentra el archivo {atp_file}")
        return

    # 1. Cargar Historial
    logging.info("Cargando base histórica (ATP + Challenger)...")
    df_hist = pd.read_csv(hist_file)
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], errors="coerce")
    logging.info(f"Base histórica cargada: {len(df_hist)} registros.")

    # 2. Cargar Maestro y CSV de 2026
    master = load_player_master()
    df_atp = pd.read_csv(atp_file)
    total_rows = len(df_atp)
    logging.info(f"Saneando un total de {total_rows} filas en {atp_file}...")

    # 3. Iterar y Recalcular (Solo las 6 columnas de rendimiento)
    for idx, row in df_atp.iterrows():
        p1, p2 = row["Jugador 1"], row["Jugador 2"]
        p1_key = str(row.get("J1 Key", "")).replace(".0", "")
        p2_key = str(row.get("J2 Key", "")).replace(".0", "")
        p1_key_clean = None if p1_key in ["", "nan", "None"] else p1_key
        p2_key_clean = None if p2_key in ["", "nan", "None"] else p2_key
        
        surf = row["Superficie"]
        # El formato en el CSV de 2026 es m/d/yy
        dt_match = pd.to_datetime(row["Fecha"], format="%m/%d/%y", errors="coerce")
        
        if pd.isna(dt_match):
            logging.warning(f"  Fila {idx} tiene fecha inválida: {row['Fecha']}")
            continue
        
        # Recalcular J1
        r1, s1 = calc_performance_refresh(p1, dt_match, surf, df_hist, master, player_key=p1_key_clean)
        u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master, player_key=p1_key_clean)
        
        # Recalcular J2
        r2, s2 = calc_performance_refresh(p2, dt_match, surf, df_hist, master, player_key=p2_key_clean)
        u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master, player_key=p2_key_clean)
        
        # Actualizar solo las columnas de rendimiento
        df_atp.at[idx, "J1 Rend. Reciente"] = r1
        df_atp.at[idx, "J1 Rend. Superficie"] = s1
        df_atp.at[idx, "Rend. Ultra reciente J1"] = u1
        df_atp.at[idx, "J2 Rend. Reciente"] = r2
        df_atp.at[idx, "J2 Rend. Superficie"] = s2
        df_atp.at[idx, "Rend. Ultra reciente J2"] = u2
        
        if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
            logging.info(f"  Progreso: {idx + 1}/{total_rows} filas procesadas.")

    # 4. Guardar archivo final
    logging.info(f"Guardando cambios en {atp_file}...")
    df_atp.to_csv(atp_file, index=False, encoding="utf-8")
    logging.info("SANEAMIENTO UNIVERSAL COMPLETADO CON ÉXITO.")

if __name__ == "__main__":
    fix_all_performance_universal()
