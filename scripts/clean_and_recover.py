import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import math
import time

sys.path.append(os.getcwd())
try:
    from scripts.refresh_data import (
        load_player_master, 
        calc_performance_refresh, 
        calc_ultra_performance_refresh,
        fetch_range,
        _get_surface,
        _winner,
        _get_points,
        get_h2h,
        _surface_map,
        get_odds_data,
        ATP_TYPES
    )
except ImportError:
    pass

def execute_rebuild():
    f_path = "data/ATP Tour 2026 Matches.csv"
    hist_f = "data/atp_challenger_fixtures_2024_2026.csv"
    
    print("Cargando backup y maestros...")
    df_atp = pd.read_csv(f_path, low_memory=False)
    
    # ── 1. PRESERVAR TODAS LAS COLUMNAS EXISTENTES ──
    # Asegurarnos de que las nuevas columnas Ultra Recientes existan
    if "Rend. Ultra reciente J1" not in df_atp.columns:
        df_atp["Rend. Ultra reciente J1"] = ""
    if "Rend. Ultra reciente J2" not in df_atp.columns:
        df_atp["Rend. Ultra reciente J2"] = ""

    # Ordenar columnas a la estructura universal que lee Streamlit (dejando Ganador al final)
    official_cols = [
        "Torneo", "Fecha", "Superficie", "Jugador 1", "J1 Puntos ATP",
        "Jugador 2", "J2 Puntos ATP", "J1 H2H", "J1 H2H %", "J2 H2H",
        "J2 H2H %", "J1 Rend. Reciente", "J1 Rend. Superficie", "Rend. Ultra reciente J1",
        "J2 Rend. Reciente", "J2 Rend. Superficie", "Rend. Ultra reciente J2",
        "Cuota J1", "Cuota J2", "Ganador"
    ]
    
    # Rellenar faltantes (Si el usuario tenía 'J1 Puntos AT' por error, etc.)
    for c in official_cols:
        if c not in df_atp.columns:
            print(f"Alerta: Columna '{c}' no existía, creada vacía.")
            df_atp[c] = ""
            
    df_atp = df_atp[official_cols].copy()
    
    # Sanear Nombres y H2H por si hay comas o basuras
    for col in df_atp.columns:
        if df_atp[col].dtype == object:
            df_atp[col] = df_atp[col].str.replace(",", ".", regex=False).str.strip()
            
    # Borrar unicamente los Rendimientos anteriores para recalcular
    cols_rend = ["J1 Rend. Reciente", "J1 Rend. Superficie", "Rend. Ultra reciente J1",
                 "J2 Rend. Reciente", "J2 Rend. Superficie", "Rend. Ultra reciente J2"]
    for c in cols_rend:
        df_atp[c] = ""

    df_hist = pd.read_csv(hist_f)
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], errors="coerce")
    
    master = load_player_master()
    smap = _surface_map()
    
    df_atp["_dt"] = pd.to_datetime(df_atp["Fecha"], errors="coerce")
    
    print(f"Recalculando rendimientos para {len(df_atp)} partidos históricos (Preservando H2H intocable)...")
    for idx, row in df_atp.iterrows():
        dt_match = row["_dt"]
        if pd.isna(dt_match): continue
        
        p1 = row["Jugador 1"]
        p2 = row["Jugador 2"]
        surf = row["Superficie"]
        
        r1, s1 = calc_performance_refresh(p1, dt_match, surf, df_hist, master)
        r2, s2 = calc_performance_refresh(p2, dt_match, surf, df_hist, master)
        u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master)
        u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master)
        
        df_atp.at[idx, "J1 Rend. Reciente"] = r1
        df_atp.at[idx, "J1 Rend. Superficie"] = s1
        df_atp.at[idx, "Rend. Ultra reciente J1"] = u1
        
        df_atp.at[idx, "J2 Rend. Reciente"] = r2
        df_atp.at[idx, "J2 Rend. Superficie"] = s2
        df_atp.at[idx, "Rend. Ultra reciente J2"] = u2

    # ── 2. BAJAR PARTIDOS FALTANTES (API) ──
    max_date = df_atp["_dt"].dropna().max()
    print(f"Fecha máxima alcanzada en histórico: {max_date.strftime('%Y-%m-%d')}")
    
    start_fetch = (max_date + timedelta(days=1))
    target_date = datetime(2026, 4, 1)
    
    new_rows = []
    if start_fetch <= target_date:
        print(f"Buscando faltantes desde {start_fetch.strftime('%Y-%m-%d')} hasta {target_date.strftime('%Y-%m-%d')}...")
        try:
            from scripts.tenis_api import get_standings
            df_standings = get_standings(event_type="ATP", save_json=False)
            pts_map = {}
            if df_standings is not None and not df_standings.empty:
                df_standings["points"] = pd.to_numeric(df_standings["points"], errors="coerce").fillna(0).astype(int)
                pts_map = df_standings.set_index("player_key")["points"].to_dict()
                for k, v in df_standings.set_index("player")["points"].to_dict().items():
                    pts_map[str(k)] = v
        except Exception as e:
            print("Error standings:", e)
            pts_map = {}
            
        cursor = start_fetch
        while cursor <= target_date:
            block_end = min(cursor + timedelta(days=6), target_date)
            start_str = cursor.strftime("%Y-%m-%d")
            end_str = block_end.strftime("%Y-%m-%d")
            print(f" -> Bloque API: {start_str} a {end_str}")
            
            chunk = fetch_range(start_str, end_str)
            odds_map = get_odds_data(start_str, end_str)
            filtered = [r for r in chunk if r.get("event_type_type") in ATP_TYPES]
            
            for f in filtered:
                m_key = str(f.get("event_key", ""))
                odd1, odd2 = odds_map.get(m_key, (0.0, 0.0))
                if odd1 == 0.0: odd1 = float(f.get("event_odd_1") or 0.0)
                if odd2 == 0.0: odd2 = float(f.get("event_odd_2") or 0.0)
                
                surf = _get_surface(f, smap)
                if "Hard" in surf or surf == "Unknown": surf = "Hard"
                if "Clay" in surf: surf = "Clay"
                if "Grass" in surf: surf = "Grass"
                
                raw_date = f.get("event_date", "")
                r_dt = pd.to_datetime(raw_date, errors="coerce")
                fecha_fmt = r_dt.strftime("%m/%d/%y") if pd.notna(r_dt) else raw_date
                
                h1_win, h1_pct, h2_win, h2_pct = 0, "0%", 0, "0%"
                p1_key, p2_key = f.get("first_player_key"), f.get("second_player_key")
                if p1_key and p2_key:
                    h2h_res = get_h2h(p1_key, p2_key, save_json=False, save_csv=False)
                    if h2h_res and "H2H" in h2h_res:
                        for m in h2h_res["H2H"]:
                            m_date = str(m.get("event_date", ""))
                            if m_date and raw_date and m_date < raw_date:
                                w = m.get("event_winner")
                                if w == "First Player": h1_win += 1
                                elif w == "Second Player": h2_win += 1
                        tot = h1_win + h2_win
                        if tot > 0:
                            h1_pct = f"{(h1_win/tot*100):.0f}%"
                            h2_pct = f"{(h2_win/tot*100):.0f}%"

                p1_name = f.get("event_first_player", "")
                p2_name = f.get("event_second_player", "")
                
                r1, s1 = calc_performance_refresh(p1_name, raw_date, surf, df_hist, master)
                u1 = calc_ultra_performance_refresh(p1_name, raw_date, df_hist, master)
                r2, s2 = calc_performance_refresh(p2_name, raw_date, surf, df_hist, master)
                u2 = calc_ultra_performance_refresh(p2_name, raw_date, df_hist, master)
                
                new_rows.append({
                    "Torneo": f.get("tournament_name", ""),
                    "Fecha": fecha_fmt,
                    "Superficie": surf,
                    "Jugador 1": p1_name,
                    "J1 Puntos ATP": _get_points(p1_name, p1_key, pts_map, master),
                    "Jugador 2": p2_name,
                    "J2 Puntos ATP": _get_points(p2_name, p2_key, pts_map, master),
                    "J1 H2H": h1_win, "J1 H2H %": h1_pct,
                    "J2 H2H": h2_win, "J2 H2H %": h2_pct,
                    "J1 Rend. Reciente": r1, "J1 Rend. Superficie": s1, "Rend. Ultra reciente J1": u1,
                    "J2 Rend. Reciente": r2, "J2 Rend. Superficie": s2, "Rend. Ultra reciente J2": u2,
                    "Cuota J1": odd1 if odd1 else "",
                    "Cuota J2": odd2 if odd2 else "",
                    "Ganador": _winner(f)
                })
                time.sleep(0.5)
                
            cursor = block_end + timedelta(days=1)

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_atp = pd.concat([df_atp, df_new], ignore_index=True)
        print(f"Se agregaron exitosamente {len(df_new)} partidos nuevos al archivo.")
        
    df_atp["Ganador"] = df_atp["Ganador"].fillna("-").replace("", "-")
    df_atp = df_atp.drop(columns=["_dt"], errors="ignore")
    
    # SALVAR DE MANERA SEGURA CON PUNTOS
    df_atp.to_csv(f_path, index=False)
    print("PROCESO COMPLETADO AL 100%. Archivo guardado, Cuotas y H2H intactos, Rendimientos actualizados.")

if __name__ == "__main__":
    execute_rebuild()
