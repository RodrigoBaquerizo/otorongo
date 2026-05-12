import pandas as pd
import os
import sys
from datetime import datetime

# Añadir el directorio actual al path para importar refresh_data
sys.path.append(os.getcwd())
try:
    from scripts.refresh_data import (
        load_player_master, 
        calc_performance_refresh
    )
except ImportError:
    from refresh_data import (
        load_player_master, 
        calc_performance_refresh
    )

def fix_surfaces_and_stats():
    tournaments_f = "data/tournaments.csv"
    challenger_hist_f = "data/atp_challenger_fixtures_2024_2026.csv"
    atp_2026_f = "data/ATP Tour 2026 Matches.csv"

    # 1. ACTUALIZAR TOURNAMENTS.CSV
    if os.path.exists(tournaments_f):
        print(f"Actualizando {tournaments_f}...")
        df_t = pd.read_csv(tournaments_f)
        df_t.loc[df_t['tournament_name'].str.contains('Miyazaki', case=False, na=False), 'tournament_sourface'] = 'Hard'
        df_t.loc[df_t['tournament_name'].str.contains('Sao Paulo', case=False, na=False), 'tournament_sourface'] = 'Clay'
        df_t.loc[df_t['tournament_name'].str.contains('Bucaramanga', case=False, na=False), 'tournament_sourface'] = 'Clay'
        df_t.to_csv(tournaments_f, index=False)


    # 3. ACTUALIZAR ATP_CHALLENGER_FIXTURES_2024_2026.CSV
    if os.path.exists(challenger_hist_f):
        print(f"Actualizando {challenger_hist_f}...")
        df_ch = pd.read_csv(challenger_hist_f)
        # Usar nombres correctos de columnas: Torneo, Superficie
        df_ch.loc[df_ch['Torneo'].str.contains('Miyazaki', case=False, na=False), 'Superficie'] = 'Hard'
        df_ch.loc[df_ch['Torneo'].str.contains('Sao Paulo', case=False, na=False), 'Superficie'] = 'Clay'
        df_ch.loc[df_ch['Torneo'].str.contains('Bucaramanga', case=False, na=False), 'Superficie'] = 'Clay'
        df_ch.to_csv(challenger_hist_f, index=False)

    # 4. RECALCULAR EN ATP TOUR 2026 MATCHES
    if os.path.exists(atp_2026_f):
        print(f"Recalculando rendimientos en {atp_2026_f}...")
        df_26 = pd.read_csv(atp_2026_f)
        
        # Corregir Superficie en este archivo también
        df_26.loc[df_26['Torneo'].str.contains('Miyazaki', case=False, na=False), 'Superficie'] = 'Hard'
        df_26.loc[df_26['Torneo'].str.contains('Sao Paulo', case=False, na=False), 'Superficie'] = 'Clay'
        df_26.loc[df_26['Torneo'].str.contains('Bucaramanga', case=False, na=False), 'Superficie'] = 'Clay'
        
        master = load_player_master()
        df_h_processed = pd.read_csv(challenger_hist_f)
        # El archivo Challenger usa formato YYYY-MM-DD nativo en Fecha
        df_h_processed["Fecha"] = pd.to_datetime(df_h_processed["Fecha"], errors="coerce")
        
        start_range = datetime(2026, 3, 24)
        end_range = datetime(2026, 3, 31)
        df_26["_dt"] = pd.to_datetime(df_26["Fecha"], format="%m/%d/%y", errors="coerce")
        
        count = 0
        for idx, row in df_26.iterrows():
            dt_match = row["_dt"]
            if pd.notna(dt_match) and start_range <= dt_match <= end_range:
                p1, p2 = row["Jugador 1"], row["Jugador 2"]
                surf = row["Superficie"]
                
                _, s1_new = calc_performance_refresh(p1, dt_match, surf, df_h_processed, master)
                _, s2_new = calc_performance_refresh(p2, dt_match, surf, df_h_processed, master)
                
                df_26.at[idx, "J1 Rend. Superficie"] = s1_new
                df_26.at[idx, "J2 Rend. Superficie"] = s2_new
                count += 1
                
        if "_dt" in df_26.columns:
            df_26 = df_26.drop(columns=["_dt"])
        df_26.to_csv(atp_2026_f, index=False)
        print(f"Hecho. Partidos actualizados: {count}")

if __name__ == "__main__":
    fix_surfaces_and_stats()
