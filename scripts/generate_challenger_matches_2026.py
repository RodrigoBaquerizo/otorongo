import pandas as pd
import os
import sys
import logging
from datetime import datetime
from bisect import bisect_right, bisect_left
import numpy as np

# Añadir el directorio raíz al path para importar scripts locales
sys.path.append(os.getcwd())

from scripts.refresh_data import (
    load_player_master
)

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

HIST_FILE = "data/atp_challenger_fixtures_2024_2026.csv"
RANKINGS_FILE = "data/atp_rankings_merged.csv"
OUTPUT_FILE = "data/Challenger Tour Matches.csv"
ELO_FILE = "data/Escala ATP - ELO.csv"

# Globales para caché
RANKINGS_BY_ID = {}
RANKINGS_BY_NAME = {}
MATCHES_BY_PLAYER = {} # {player_id/name: indices_en_df_hist}
DF_HIST_CLEAN = None
ELO_SCALE = None

def prepare_rankings(df_rankings):
    global RANKINGS_BY_ID, RANKINGS_BY_NAME
    if df_rankings.empty: return
    logging.info("Preparando caché de rankings...")
    if "player_id" in df_rankings.columns:
        for pid, group in df_rankings.dropna(subset=["player_id"]).groupby("player_id"):
            sorted_group = group.sort_values("date")
            RANKINGS_BY_ID[str(int(pid))] = {"dates": sorted_group["date"].tolist(), "points": sorted_group["points"].tolist()}
    for name, group in df_rankings.groupby("player_name"):
        sorted_group = group.sort_values("date")
        RANKINGS_BY_NAME[name.lower().strip()] = {"dates": sorted_group["date"].tolist(), "points": sorted_group["points"].tolist()}

def prepare_history(df_hist, master):
    global DF_HIST_CLEAN, MATCHES_BY_PLAYER
    logging.info("Preparando historial optimizado...")
    df = df_hist.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"]).sort_values("Fecha")
    
    # Normalización de strings una sola vez
    df["p1_norm"] = df["Jugador 1"].str.lower().str.strip()
    df["p2_norm"] = df["Jugador 2"].str.lower().str.strip()
    df["win_norm"] = df["Ganador"].str.lower().str.strip()
    df["k1_clean"] = df["J1 Key"].astype(str).str.replace(".0", "").replace("nan", "")
    df["k2_clean"] = df["J2 Key"].astype(str).str.replace(".0", "").replace("nan", "")
    
    # Pre-normalización de superficie
    df["surf_norm"] = df.apply(lambda r: normalize_surface_string(r["Superficie"], r["Torneo"]), axis=1)
    
    # Mapeo de jugadores a índices
    for idx, row in df.iterrows():
        p1n, p2n = row["p1_norm"], row["p2_norm"]
        k1, k2 = row["k1_clean"], row["k2_clean"]
        
        matches_keys = [p1n, p2n]
        if k1: matches_keys.append(k1)
        if k2: matches_keys.append(k2)
        
        for k in set(matches_keys):
            if k not in MATCHES_BY_PLAYER: MATCHES_BY_PLAYER[k] = []
            MATCHES_BY_PLAYER[k].append(idx)
            
    DF_HIST_CLEAN = df
    logging.info("Historial optimizado completado.")

def get_points_fast(player_name, player_key, match_date, master):
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    res = RANKINGS_BY_ID.get(pk_str)
    if not res:
        search_names = {player_name.lower().strip()}
        if pk_str and pk_str in master["by_key"]:
            search_names.add(master["by_key"][pk_str]["canonical_name"].lower().strip())
            for a in master["by_key"][pk_str]["aliases"]: search_names.add(a.lower().strip())
        for n in search_names:
            if n in RANKINGS_BY_NAME:
                res = RANKINGS_BY_NAME[n]
                break
    if not res: return 0
    dates = res["dates"]
    idx = bisect_right(dates, match_date)
    return res["points"][idx-1] if idx > 0 else res["points"][0]

def normalize_surface_string(surface, torneo=""):
    if pd.isna(surface): return "Desconocida"
    s = str(surface).lower()
    t = str(torneo).lower()
    
    # Overrides específicos
    if "davis" in t: return "Hard"
    if t == "brasilia 2": return "Clay"
    if t in ["fujairah", "metepec"]: return "Hard"
    
    if "hard" in s: return "Hard"
    if "clay" in s: return "Clay"
    if "grass" in s: return "Grass"
    return surface.capitalize()

def calc_perf_fast(player_name, player_key, match_date, surface, torneo, master, months=12):
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    
    # Identificar todas las formas del jugador
    p_keys = {player_name.lower().strip()}
    if pk_str: p_keys.add(pk_str)
    if pk_str and pk_str in master["by_key"]:
        p_keys.add(master["by_key"][pk_str]["canonical_name"].lower().strip())
        for a in master["by_key"][pk_str]["aliases"]: p_keys.add(a.lower().strip())
        
    # Obtener índices de partidos relevantes (desde caché)
    relevant_indices = set()
    for k in p_keys:
        if k in MATCHES_BY_PLAYER:
            relevant_indices.update(MATCHES_BY_PLAYER[k])
            
    if not relevant_indices: return "0%", "0%"
    
    df_p = DF_HIST_CLEAN.loc[list(relevant_indices)]
    
    # Filtro de fecha (12 meses)
    period_start = match_date - pd.DateOffset(months=months)
    mask_date = (df_p["Fecha"] >= period_start) & (df_p["Fecha"] < match_date)
    df_period = df_p[mask_date]
    
    def get_pct(df_sub):
        if df_sub.empty: return "0%"
        wins = len(df_sub[df_sub["win_norm"].isin(p_keys)])
        return f"{(wins/len(df_sub)*100):.0f}%"

    # Recent Performance
    recent_pct = get_pct(df_period)
    
    # Surface Performance
    surf_norm = normalize_surface_string(surface, torneo)
    df_surf = df_period[df_period["surf_norm"] == surf_norm]
    surf_pct = get_pct(df_surf)
    
    return recent_pct, surf_pct

def calc_ultra_fast(player_name, player_key, match_date, master):
    global ELO_SCALE
    if ELO_SCALE is None:
        try:
            df_s = pd.read_csv(ELO_FILE)
            # Corregir lógica: Convertir a lista de tuplas (límite, valor) y ordenar descendente
            # El archivo tiene columnas "Hasta (puntos)" y "Valor"
            ELO_SCALE = list(df_s.itertuples(index=False, name=None))
            ELO_SCALE.sort(key=lambda x: x[0], reverse=True)
        except Exception as e:
            logging.error(f"Error cargando ELO_FILE: {e}")
            ELO_SCALE = []
        
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    p_keys = {player_name.lower().strip()}
    if pk_str: p_keys.add(pk_str)
    if pk_str and pk_str in master["by_key"]:
        p_keys.add(master["by_key"][pk_str]["canonical_name"].lower().strip())
        for a in master["by_key"][pk_str]["aliases"]: p_keys.add(a.lower().strip())
        
    relevant_indices = set()
    for k in p_keys:
        if k in MATCHES_BY_PLAYER: relevant_indices.update(MATCHES_BY_PLAYER[k])
    
    if not relevant_indices: return "0%"
    
    df_p = DF_HIST_CLEAN.loc[list(relevant_indices)]
    period_start = match_date - pd.DateOffset(days=30)
    df_ultra = df_p[(df_p["Fecha"] >= period_start) & (df_p["Fecha"] < match_date)]
    if df_ultra.empty: return "0%"
    
    def get_scale_value(points):
        for pts_limit, val in ELO_SCALE:
            if points >= pts_limit: return val
        return 40

    total_score = 0
    total_weight = 0
    
    for _, m in df_ultra.iterrows():
        # Identificar oponente
        is_p1 = (m["p1_norm"] in p_keys or m["k1_clean"] in p_keys)
        opp_name = m["Jugador 2"] if is_p1 else m["Jugador 1"]
        opp_key = m["J2 Key"] if is_p1 else m["J1 Key"]
        
        # Puntos oponente en ese momento
        opp_pts = get_points_fast(opp_name, opp_key, m["Fecha"], master)
        
        # Buscar ELO
        try:
            points_val = int(float(opp_pts)) if opp_pts not in ["", None, "nan"] else 0
        except:
            points_val = 0
            
        weight = get_scale_value(points_val)
        
        won = m["win_norm"] in p_keys
        total_score += (1 if won else 0) * weight
        total_weight += weight
        
    return f"{(total_score/total_weight*100):.0f}%" if total_weight > 0 else "0%"

def calculate_local_h2h(p1_key, p1_name, p2_key, p2_name, match_date, master):
    k1 = str(p1_key).replace(".0", "") if p1_key and str(p1_key) != "nan" else None
    k2 = str(p2_key).replace(".0", "") if p2_key and str(p2_key) != "nan" else None
    n1, n2 = p1_name.lower().strip(), p2_name.lower().strip()
    
    keys1, keys2 = {n1}, {n2}
    if k1: keys1.add(k1)
    if k2: keys2.add(k2)
    # Alias
    if k1 and k1 in master["by_key"]:
        keys1.add(master["by_key"][k1]["canonical_name"].lower().strip())
        for a in master["by_key"][k1]["aliases"]: keys1.add(a.lower().strip())
    if k2 and k2 in master["by_key"]:
        keys2.add(master["by_key"][k2]["canonical_name"].lower().strip())
        for a in master["by_key"][k2]["aliases"]: keys2.add(a.lower().strip())
        
    # Obtener índices donde juegan ambos
    idx1 = set()
    for k in keys1:
        if k in MATCHES_BY_PLAYER: idx1.update(MATCHES_BY_PLAYER[k])
    idx2 = set()
    for k in keys2:
        if k in MATCHES_BY_PLAYER: idx2.update(MATCHES_BY_PLAYER[k])
        
    common = idx1.intersection(idx2)
    if not common: return 0, "0%", 0, "0%"
    
    df_h2h = DF_HIST_CLEAN.loc[list(common)]
    df_h2h = df_h2h[df_h2h["Fecha"] < match_date]
    
    h1_win, h2_win = 0, 0
    for _, m in df_h2h.iterrows():
        if m["win_norm"] in keys1: h1_win += 1
        elif m["win_norm"] in keys2: h2_win += 1
        
    total = h1_win + h2_win
    h1_pct = f"{(h1_win/total*100):.0f}%" if total > 0 else "0%"
    h2_pct = f"{(h2_win/total*100):.0f}%" if total > 0 else "0%"
    return h1_win, h1_pct, h2_win, h2_pct

def generate():
    logging.info("Iniciando generación MEGA-OPTIMIZADA...")
    df_hist_raw = pd.read_csv(HIST_FILE)
    df_rankings = pd.read_csv(RANKINGS_FILE) if os.path.exists(RANKINGS_FILE) else pd.DataFrame()
    if not df_rankings.empty:
        df_rankings["date"] = pd.to_datetime(df_rankings["date"], errors="coerce")
    
    master = load_player_master()
    prepare_rankings(df_rankings)
    prepare_history(df_hist_raw, master)
    
    # Se excluyen los torneos que ya están en el archivo principal de ATP 2026
    atp_tournaments = set()
    if os.path.exists("data/ATP Tour 2026 Matches.csv"):
        df_atp = pd.read_csv("data/ATP Tour 2026 Matches.csv")
        atp_tournaments = set(df_atp["Torneo"].dropna().unique())
        
    df_2026 = DF_HIST_CLEAN[ (DF_HIST_CLEAN["Fecha"].dt.year == 2026) & (~DF_HIST_CLEAN["Torneo"].isin(atp_tournaments)) ].copy()
    
    # Filtro extra por palabras clave típicas de ATP si aún quedan
    atp_keywords = ["united cup", "brisbane", "hong kong", "adelaide", "auckland"]
    for kw in atp_keywords:
        df_2026 = df_2026[~df_2026["Torneo"].str.lower().str.contains(kw, na=False)]
        
    total = len(df_2026)
    logging.info(f"Procesando {total} partidos Challenger...")
    
    results = []
    for idx, row in df_2026.iterrows():
        p1, p2 = row["Jugador 1"], row["Jugador 2"]
        k1, k2 = row["k1_clean"], row["k2_clean"]
        surf, dt = row["Superficie"], row["Fecha"]
        
        pts1 = get_points_fast(p1, k1, dt, master)
        pts2 = get_points_fast(p2, k2, dt, master)
        h1w, h1p, h2w, h2p = calculate_local_h2h(k1, p1, k2, p2, dt, master)
        r1, s1 = calc_perf_fast(p1, k1, dt, surf, row["Torneo"], master)
        r2, s2 = calc_perf_fast(p2, k2, dt, surf, row["Torneo"], master)
        u1 = calc_ultra_fast(p1, k1, dt, master)
        u2 = calc_ultra_fast(p2, k2, dt, master)
        
        results.append({
            "Torneo": row["Torneo"], "Fecha": dt.strftime("%-m/%-d/%y"), "Superficie": surf,
            "Jugador 1": p1, "J1 Key": k1, "J1 Puntos ATP": pts1,
            "Jugador 2": p2, "J2 Key": k2, "J2 Puntos ATP": pts2,
            "J1 H2H": h1w, "J1 H2H %": h1p, "J2 H2H": h2w, "J2 H2H %": h2p,
            "J1 Rend. Reciente": r1, "J1 Rend. Superficie": s1, "Rend. Ultra reciente J1": u1,
            "J2 Rend. Reciente": r2, "J2 Rend. Superficie": s2, "Rend. Ultra reciente J2": u2,
            "Cuota J1": row.get("Cuota J1", ""), "Cuota J2": row.get("Cuota J2", ""),
            "Ganador": row["Ganador"], "ID Partido": row.get("ID Partido", "")
        })
        if (len(results)) % 500 == 0 or len(results) == total:
            logging.info(f"Progreso: {len(results)}/{total}")
            
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    logging.info(f"Completado: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate()
