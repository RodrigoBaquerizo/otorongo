import pandas as pd
import os
import json
import logging
from datetime import datetime
import shutil

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Rutas de archivos
CHA_FILE = "data/Challenger Tour Matches.csv"
HIST_FILE = "data/atp_challenger_fixtures_2024_2026.csv"
PLAYER_MASTER = "data/player_master.json"

# Importar funciones de lógica desde refresh_data
try:
    from scripts.refresh_data import (
        calc_performance_refresh,
        calc_ultra_performance_refresh,
        _load_stats_resources,
        load_player_master
    )
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from scripts.refresh_data import (
        calc_performance_refresh,
        calc_ultra_performance_refresh,
        _load_stats_resources,
        load_player_master
    )

def backup_file(filepath):
    if os.path.exists(filepath):
        backup_path = f"{filepath}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        logging.info(f"Backup creado en: {backup_path}")
        return backup_path
    return None

def run_recalculation():
    logging.info("Iniciando recalculación masiva de estadísticas Challenger...")
    
    # 1. Realizar Backup
    backup_file(CHA_FILE)
    
    # 2. Cargar Recursos
    logging.info("Cargando recursos (Histórico, Rankings, Maestro)...")
    df_cha = pd.read_csv(CHA_FILE)
    df_hist = pd.read_csv(HIST_FILE)
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], format='mixed', dayfirst=True)
    
    master = load_player_master()
    _load_stats_resources() # Solo para inicializar variables si fuera necesario, aunque ya se hace en calc_ultra
    
    total_rows = len(df_cha)
    logging.info(f"Procesando {total_rows} partidos...")
    
    # 3. Iterar y Recalcular
    processed_count = 0
    START_INDEX = 0 # Cambiar si se desea reanudar desde una fila específica
    MAX_ROWS = None # Cambiar a None para proceso completo
    
    for idx, row in df_cha.iterrows():
        if idx < START_INDEX:
            continue
        if MAX_ROWS and processed_count >= MAX_ROWS:
            break
            
        # Determinar la fecha de forma robusta
        f_dt = str(row.get("Fecha_dt", "")).strip()
        if f_dt and f_dt != "nan" and f_dt != "":
            try:
                match_date = pd.to_datetime(f_dt)
            except:
                match_date = pd.to_datetime(row["Fecha"], format='%m/%d/%y', errors='coerce')
        else:
            match_date = pd.to_datetime(row["Fecha"], format='%m/%d/%y', errors='coerce')
            
        if pd.isna(match_date):
            logging.warning(f"Fila {idx}: No se pudo determinar la fecha para {row['Torneo']}. Saltando...")
            continue
            
        surface = row["Superficie"]
        p1_name, p1_key = row["Jugador 1"], row["J1 Key"]
        p2_name, p2_key = row["Jugador 2"], row["J2 Key"]
        
        # Rendimiento J1
        rec1, surf1 = calc_performance_refresh(p1_name, match_date, surface, df_hist, master, player_key=p1_key)
        ult1 = calc_ultra_performance_refresh(p1_name, match_date, df_hist, master, player_key=p1_key)
        
        # Rendimiento J2
        rec2, surf2 = calc_performance_refresh(p2_name, match_date, surface, df_hist, master, player_key=p2_key)
        ult2 = calc_ultra_performance_refresh(p2_name, match_date, df_hist, master, player_key=p2_key)
        
        # Actualizar DataFrame
        df_cha.at[idx, "J1 Rend. Reciente"] = rec1
        df_cha.at[idx, "J1 Rend. Superficie"] = surf1
        df_cha.at[idx, "Rend. Ultra reciente J1"] = ult1
        
        df_cha.at[idx, "J2 Rend. Reciente"] = rec2
        df_cha.at[idx, "J2 Rend. Superficie"] = surf2
        df_cha.at[idx, "Rend. Ultra reciente J2"] = ult2
        
        # Aprovechar para normalizar Fecha_dt si estaba vacía
        if not f_dt or f_dt == "nan":
            df_cha.at[idx, "Fecha_dt"] = match_date.strftime("%Y-%m-%d")
        
        processed_count += 1
        if processed_count % 100 == 0:
            logging.info(f"Progreso: {processed_count}/{total_rows if not MAX_ROWS else MAX_ROWS}...")
            # Auto-guardado parcial para evitar pérdida por interrupción
            df_cha.to_csv(CHA_FILE, index=False)
            logging.info("Guardado parcial realizado.")

    # 4. Guardar resultados finales
    if MAX_ROWS:
        logging.info(f"MODO PRUEBA: Guardando solo {MAX_ROWS} filas en {CHA_FILE} para validación.")
    else:
        logging.info(f"Guardando cambios finales en {CHA_FILE}...")
    
    df_cha.to_csv(CHA_FILE, index=False)
    logging.info("✅ Recalculación completada con éxito.")

if __name__ == "__main__":
    run_recalculation()
