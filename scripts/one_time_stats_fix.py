import pandas as pd
import os
import sys
from datetime import datetime

# Añadir el directorio actual al path para importar refresh_data
sys.path.append(os.getcwd())
try:
    from scripts.refresh_data import (
        load_player_master, 
        calc_performance_refresh, 
        calc_ultra_performance_refresh
    )
except ImportError:
    # Si no se puede importar como scripts.refresh_data, probar directo
    from refresh_data import (
        load_player_master, 
        calc_performance_refresh, 
        calc_ultra_performance_refresh
    )

def run_one_time_fix():
    atp_f = "data/ATP Tour 2026 Matches.csv"
    hist_f = "data/atp_challenger_fixtures_2024_2026.csv"
    
    if not os.path.exists(atp_f) or not os.path.exists(hist_f):
        print("Error: No se encuentran los archivos CSV necesarios.")
        return

    # 1. Cargar Datos
    print("Cargando archivos y maestros...")
    df_atp = pd.read_csv(atp_f)
    print(f"Cargado {atp_f}. Filas={len(df_atp)}")
    
    df_hist = pd.read_csv(hist_f)
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], format="%m/%d/%y", errors="coerce")
    
    master = load_player_master()
    
    # Rango para Rendimiento Reciente/Superficie (24/03/26 al 31/03/26)
    start_range = datetime(2026, 3, 24)
    end_range = datetime(2026, 3, 31)
    
    # Temporales para cálculos
    df_atp["_dt"] = pd.to_datetime(df_atp["Fecha"], format="%m/%d/%y", errors="coerce")
    
    total_rows = len(df_atp)
    print(f"Procesando {total_rows} filas...")

    c_ultra = 0
    c_recent = 0

    for idx, row in df_atp.iterrows():
        p1, p2 = row["Jugador 1"], row["Jugador 2"]
        surf = row["Superficie"]
        dt_match = row["_dt"]
        
        if pd.isna(dt_match):
            continue
            
        # REGLA 1: Siempre actualizar Ultra Reciente
        u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master)
        u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master)
        df_atp.at[idx, "Rend. Ultra reciente J1"] = u1
        df_atp.at[idx, "Rend. Ultra reciente J2"] = u2
        c_ultra += 1
        
        # REGLA 2: Actualizar Reciente y Superficie solo en el rango 24-31 de marzo
        if start_range <= dt_match <= end_range:
            r1, s1 = calc_performance_refresh(p1, dt_match, surf, df_hist, master)
            r2, s2 = calc_performance_refresh(p2, dt_match, surf, df_hist, master)
            
            df_atp.at[idx, "J1 Rend. Reciente"] = r1
            df_atp.at[idx, "J1 Rend. Superficie"] = s1
            df_atp.at[idx, "J2 Rend. Reciente"] = r2
            df_atp.at[idx, "J2 Rend. Superficie"] = s2
            c_recent += 1

        if idx > 0 and (idx % 100 == 0 or idx == total_rows-1):
            print(f"  Progreso: {idx+1}/{total_rows}...")

    # Limpiar y Guardar
    if "_dt" in df_atp.columns:
        df_atp = df_atp.drop(columns=["_dt"])
    
    df_atp.to_csv(atp_f, index=False)
    
    print("\n--- RESUMEN DE ACTUALIZACIÓN ÚNICA ---")
    print(f"Filas con Ultra Reciente actualizado: {c_ultra}")
    print(f"Filas con Reciente/Superficie actualizado (marzo 24-31): {c_recent}")
    print(f"Archivo guardado en {atp_f}")

if __name__ == "__main__":
    run_one_time_fix()
