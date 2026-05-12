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
try:
    from scripts.refresh_data import (
        calc_ultra_performance_refresh, 
        load_player_master
    )
except ImportError:
    logging.error("Error al importar scripts.refresh_data")
    sys.exit(1)

def recalculate_ultra_universal_2026():
    """
    Recálculo Universal de Rendimiento Ultra Reciente para todo el archivo 2026.
    Aprovecha la armonización de IDs y Puntos ATP reciente.
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
    logging.info(f"Recalculando Rendimiento Ultra en {total_rows} filas...")

    # 3. Iterar y Recalcular
    for idx, row in df_atp.iterrows():
        p1, p2 = row["Jugador 1"], row["Jugador 2"]
        p1_key = str(row.get("J1 Key", "")).replace(".0", "")
        p2_key = str(row.get("J2 Key", "")).replace(".0", "")
        p1_key_clean = None if p1_key in ["", "nan", "None"] else p1_key
        p2_key_clean = None if p2_key in ["", "nan", "None"] else p2_key
        
        # Fecha en el CSV es m/d/yy
        dt_match = pd.to_datetime(row["Fecha"], format="%m/%d/%y", errors="coerce")
        
        if pd.isna(dt_match):
            logging.warning(f"  Fila {idx} tiene fecha inválida: {row['Fecha']}")
            continue
        
        # Recalcular Rendimiento Ultra J1
        u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master, player_key=p1_key_clean)
        
        # Recalcular Rendimiento Ultra J2
        u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master, player_key=p2_key_clean)
        
        # Actualizar las dos columnas específicas
        df_atp.at[idx, "Rend. Ultra reciente J1"] = u1
        df_atp.at[idx, "Rend. Ultra reciente J2"] = u2
        
        if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
            logging.info(f"  Progreso: {idx + 1}/{total_rows} filas procesadas.")

    # 4. Guardar archivo final
    logging.info(f"Guardando cambios en {atp_file}...")
    df_atp.to_csv(atp_file, index=False, encoding="utf-8")
    logging.info("RECÁLCULO UNIVERSAL DE RENDIMIENTO ULTRA COMPLETADO.")

if __name__ == "__main__":
    recalculate_ultra_universal_2026()
