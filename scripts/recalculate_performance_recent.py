import pandas as pd
import json
import os
import sys
from datetime import datetime

sys.path.append(os.getcwd())
from scripts.refresh_data import calc_performance_refresh, calc_ultra_performance_refresh

def recalculate_last_days():
    ATP_FILE = 'data/ATP Tour 2026 Matches.csv'
    HIST_FILE = 'data/atp_challenger_fixtures_2024_2026.csv'
    MASTER_JSON = 'data/player_master.json'

    if not os.path.exists(ATP_FILE):
        print(f'Error: {ATP_FILE} not found.')
        return

    df_atp = pd.read_csv(ATP_FILE)
    # Ensure date format for comparison
    df_atp['Fecha_dt'] = pd.to_datetime(df_atp['Fecha'], errors='coerce')
    
    # Filter for dates >= 2026-03-24
    mask = df_atp['Fecha_dt'] >= '2026-03-24'
    df_recent = df_atp[mask].copy()
    
    if df_recent.empty:
        print('No matches found from 2026-03-24 onwards.')
        return

    print(f'Recalculating performance for {len(df_recent)} matches...')

    # Load resources
    df_hist = pd.read_csv(HIST_FILE)
    df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'], errors='coerce')
    
    with open(MASTER_JSON, 'r') as f:
        master = json.load(f)

    for idx, row in df_recent.iterrows():
        dt_match = row['Fecha_dt']
        surf = row['Superficie']
        p1 = row['Jugador 1']
        p2 = row['Jugador 2']
        p1_key = str(row.get('J1 Key', '')).replace('.0', '')
        p2_key = str(row.get('J2 Key', '')).replace('.0', '')
        
        # J1
        r1, s1 = calc_performance_refresh(p1, dt_match, surf, df_hist, master, player_key=p1_key)
        u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master, player_key=p1_key)
        
        # J2
        r2, s2 = calc_performance_refresh(p2, dt_match, surf, df_hist, master, player_key=p2_key)
        u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master, player_key=p2_key)
        
        # Update columns
        df_atp.at[idx, 'J1 Rend. Reciente'] = r1
        df_atp.at[idx, 'J1 Rend. Superficie'] = s1
        df_atp.at[idx, 'Rend. Ultra reciente J1'] = u1
        
        df_atp.at[idx, 'J2 Rend. Reciente'] = r2
        df_atp.at[idx, 'J2 Rend. Superficie'] = s2
        df_atp.at[idx, 'Rend. Ultra reciente J2'] = u2

    # Drop temp column and save
    df_atp = df_atp.drop(columns=['Fecha_dt'])
    df_atp.to_csv(ATP_FILE, index=False, encoding='utf-8')
    print('Recalculation complete!')

if __name__ == '__main__':
    recalculate_last_days()
