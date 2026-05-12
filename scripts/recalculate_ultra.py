import pandas as pd
import os
import logging
from datetime import datetime, timedelta
from refresh_data import (
    calc_ultra_performance_refresh, 
    _load_stats_resources, 
    HIST_FILE, 
    ATP26_FILE
)

def recalculate_ultra():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    if not os.path.exists(ATP26_FILE):
        print(f"Error: {ATP26_FILE} no existe.")
        return

    print("Cargando datos...")
    df_atp = pd.read_csv(ATP26_FILE)
    df_hist = pd.read_csv(HIST_FILE)
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"])
    
    # Asegurar que los recursos se carguen una vez
    _load_stats_resources()
    
    total = len(df_atp)
    print(f"Recalculando Rendimiento Ultra Reciente (v2) para {total} filas...")
    
    for idx, row in df_atp.iterrows():
        p1, p2 = row["Jugador 1"], row["Jugador 2"]
        
        # Convertir fecha m/d/yy a datetime
        try:
            dt_match = pd.to_datetime(row["Fecha"], format="%m/%d/%y")
        except:
            continue
            
        # Calcular nueva métrica
        u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist)
        u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist)
        
        # Actualizar
        df_atp.at[idx, "Rend. Ultra reciente J1"] = u1
        df_atp.at[idx, "Rend. Ultra reciente J2"] = u2
        
        if (idx + 1) % 50 == 0:
            print(f"Progreso: {idx+1}/{total}...")

    # Guardar cambios
    df_atp.to_csv(ATP26_FILE, index=False, encoding="utf-8")
    print(f"\n✅ Recalculación completada y guardada en {ATP26_FILE}")

if __name__ == "__main__":
    recalculate_ultra()
