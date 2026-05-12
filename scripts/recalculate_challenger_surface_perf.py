import pandas as pd
import os
import sys
import logging
from datetime import datetime

# Añadir el directorio raíz al path para importar scripts locales
sys.path.append(os.getcwd())

# Reutilizamos las funciones optimizadas de nuestro script anterior
import scripts.generate_challenger_matches_2026 as gen

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HIST_FILE = "data/atp_challenger_fixtures_2024_2026.csv"
TARGET_FILE = "data/Challenger Tour Matches.csv"

def recalculate_surface_perf():
    logging.info("Iniciando recálculo de rendimiento de superficie (Tarea única)...")
    
    if not os.path.exists(TARGET_FILE):
        logging.error(f"No se encuentra {TARGET_FILE}. Nada que recalcular.")
        return

    # 1. Cargar Maestro y Historial Correcto
    master = gen.load_player_master()
    df_hist_raw = pd.read_csv(HIST_FILE)
    gen.prepare_history(df_hist_raw, master)
    
    # 2. Cargar archivo objetivo
    df_target = pd.read_csv(TARGET_FILE)
    df_target["Fecha_dt"] = pd.to_datetime(df_target["Fecha"], format="%m/%d/%y", errors="coerce")
    
    total = len(df_target)
    logging.info(f"Recalculando {total} partidos en {TARGET_FILE}...")
    
    # 3. Iterar y actualizar
    # Vamos a usar una copia del historial para buscar la superficie oficial si el match tiene ID
    hist_by_id = {}
    if "ID Partido" in df_hist_raw.columns:
        # Usar el último valor por ID por si acaso hay duplicados
        hist_by_id = df_hist_raw.dropna(subset=["ID Partido"]).set_index("ID Partido")["Superficie"].to_dict()

    for idx, row in df_target.iterrows():
        p1, k1 = row["Jugador 1"], str(row.get("J1 Key", "")).replace(".0", "")
        p2, k2 = row["Jugador 2"], str(row.get("J2 Key", "")).replace(".0", "")
        dt = row["Fecha_dt"]
        match_id = row.get("ID Partido")
        
        # Determinar superficie "verdadera"
        official_surf = row["Superficie"]
        if match_id and not pd.isna(match_id) and str(match_id) in hist_by_id:
            official_surf = hist_by_id[str(match_id)]
        
        # Siempre normalizamos para el cálculo
        k1_clean = None if k1 in ["", "nan", "None"] else k1
        k2_clean = None if k2 in ["", "nan", "None"] else k2
        
        # Recalcular Rend. Superficie para ambos jugadores
        _, s1 = gen.calc_perf_fast(p1, k1_clean, dt, official_surf, row["Torneo"], master)
        _, s2 = gen.calc_perf_fast(p2, k2_clean, dt, official_surf, row["Torneo"], master)
        
        # Actualizar filas
        df_target.at[idx, "Superficie"] = official_surf
        df_target.at[idx, "J1 Rend. Superficie"] = s1
        df_target.at[idx, "J2 Rend. Superficie"] = s2
        
        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            logging.info(f"Progreso: {idx + 1}/{total}")

    pd_df = pd.DataFrame(df_target)
    pd_df.to_csv(TARGET_FILE, index=False, encoding="utf-8")
    logging.info(f"SANEAMIENTO FINALIZADO. Se han actualizado las superficies y rendimientos en {TARGET_FILE}.")

if __name__ == "__main__":
    recalculate_surface_perf()
