"""
refresh_data.py

Actualización incremental de datos de tenis.
Descarga solo los días ausentes y actualiza dos archivos:
  1. data/atp_challenger_fixtures_2024_2026.csv  (ATP + Challenger Singles)
  2. data/ATP Tour 2026 Matches.csv              (ATP Singles con cuotas)

Estrategia:
  - Leer la fecha máxima de cada archivo.
  - Descargar solo desde ese día hasta ayer.
  - Fusionar y deduplicar.
"""

import pandas as pd
import requests
import os
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import unicodedata
import re

# Importar funciones de la API local
from scripts.tenis_api import get_standings, get_h2h, get_odds_data, safe_get

load_dotenv(override=True)
from scripts.data_manager import DataManager

_manager_instance = None
def _get_manager():
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DataManager()
    return _manager_instance

API_KEY = os.getenv("API_KEY")
BASE_URL = "https://api.api-tennis.com/tennis/?method=get_fixtures"
RANKINGS_FILE = "data/atp_rankings_merged.csv"

HIST_FILE    = "data/atp_challenger_fixtures_2024_2026.csv"
ATP26_FILE   = "data/ATP Tour 2026 Matches.csv"
CHA26_FILE   = "data/Challenger Tour Matches.csv"
TRN_FILE     = "data/tournaments.csv"

# Tipos de eventos a incluir (con sus variantes)
ATP_TYPES       = {"Atp Singles"}
CHA_TYPES       = {"Challenger Men Singles", "Challenger Men - Singles"}
HIST_TYPES      = {"Atp Singles", "Challenger Men Singles", "Challenger Men - Singles"}


def _surface_map():
    df = _get_manager().load_table("tournaments")
    if df.empty or "tournament_key" not in df.columns:
        return {}
    return df.set_index("tournament_key")["tournament_sourface"].to_dict()


def _normalize_surface(df: pd.DataFrame) -> pd.DataFrame:
    cond = [
        df["Superficie"].str.contains("Hard",  case=False, na=False),
        df["Superficie"].str.contains("Clay",  case=False, na=False),
        df["Superficie"].str.contains("Grass", case=False, na=False),
    ]
    df["Superficie"] = np.select(cond, ["Hard", "Clay", "Grass"], default=df["Superficie"])
    df.loc[df["Torneo"].str.contains("Davis", case=False, na=False), "Superficie"] = "Hard"
    df.loc[df["Torneo"] == "Brasilia 2",  "Superficie"] = "Clay"
    df.loc[df["Torneo"] == "Fujairah",    "Superficie"] = "Hard"
    df.loc[df["Torneo"] == "Metepec",     "Superficie"] = "Hard"
    return df


def fetch_range(from_date: str, to_date: str) -> list:
    url = f"{BASE_URL}&APIkey={API_KEY}&date_start={from_date}&date_stop={to_date}"
    try:
        r = safe_get(url)
        if r.status_code == 200:
            data = r.json()
            if data.get("success") == 1:
                return data.get("result", [])
    except Exception as e:
        logging.error(f"Error crítico en fetch_range: {e}")
    return []


def _get_surface(fixture: dict, smap: dict) -> str:
    t_key = fixture.get("tournament_key")
    try:
        k = int(t_key) if t_key and str(t_key).isdigit() else t_key
        return smap.get(k, "Unknown")
    except Exception:
        return "Unknown"


def _winner(f: dict) -> str:
    status = str(f.get("event_status", "")).lower()
    if "cancelled" in status:
        return "Cancelado"
    if "retired" in status:
        return "Retirado"
        
    if f.get("event_winner") == "First Player":
        return f.get("event_first_player", "-")
    if f.get("event_winner") == "Second Player":
        return f.get("event_second_player", "-")
    return "-"


def normalize_name_robust(name: str) -> str:
    """Standardizes names: lowercase, no accents, uniform spacing for initials."""
    if not name or pd.isna(name) or str(name).strip() == "":
        return ""
    # 1. Lowercase and remove accents
    s = str(name).strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    # 2. Standardize dots (J.M. -> j. m.)
    s = s.replace(".", ". ")
    # 3. Clean extra spaces and non-alpha chars (keep dots for initials)
    s = re.sub(r"[^a-z0-9\s.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_name(name: str) -> str:
    """LEGACY: Extracts last name. Use normalize_name_robust for keys."""
    if not name or pd.isna(name): return ""
    name = str(name).lower().strip()
    if ". " in name:
        return name.split(". ")[-1]
    if " " in name:
        return name.split(" ")[-1]
    return name


def load_player_master():
    path_json = "data/player_master.json"
    path_csv = "data/master-normalizacion-players-key.csv"
    
    master = {"by_key": {}, "by_alias": {}}
    
    # 1. Cargar JSON (respaldo/histórico)
    if os.path.exists(path_json):
        try:
            with open(path_json, "r") as f:
                master = json.load(f)
        except Exception as e:
            logging.error(f"Error cargando player_master.json: {e}")

    # 2. Cargar CSV (Maestro Principal proporcionado por usuario)
    if os.path.exists(path_csv):
        try:
            df_m = pd.read_csv(path_csv)
            for _, row in df_m.iterrows():
                p_name = str(row.get("player_name", "")).strip()
                p_fullname = str(row.get("player_full_name", "")).strip()
                p_key = str(int(float(row.get("player_key", 0)))) if pd.notna(row.get("player_key")) else None
                ppn = str(row.get("ppn", "")).strip()
                
                if not p_key: continue

                # Versiones robustas para el diccionario de alias
                aliases_to_register = []
                if p_name: aliases_to_register.append(p_name)
                if p_fullname: aliases_to_register.append(p_fullname)
                if ppn: aliases_to_register.append(ppn)
                
                for a in aliases_to_register:
                    # Registrar versión literal y robusta
                    master["by_alias"][a.lower().strip()] = p_key
                    master["by_alias"][normalize_name_robust(a)] = p_key
                        
                # Registro en by_key
                if p_key not in master["by_key"]:
                    master["by_key"][p_key] = {"canonical_name": p_name, "aliases": list(set(aliases_to_register))}
                else:
                    existing_aliases = set(master["by_key"][p_key]["aliases"])
                    master["by_key"][p_key]["aliases"] = list(existing_aliases.union(set(aliases_to_register)))
        except Exception as e:
            logging.error(f"Error cargando maestro CSV: {e}")
            
    return master

def save_player_master(master):
    path = "data/player_master.json"
    with open(path, "w") as f:
        json.dump(master, f, indent=2)

def update_player_master(master, p_key, p_name):
    """Aprende nuevas asociaciones de nombre -> ID de forma automática"""
    if not p_key or not p_name: return
    p_key = str(p_key)
    norm = p_name.lower().strip()
    
    # 1. Registrar por ID
    if p_key not in master["by_key"]:
        master["by_key"][p_key] = {"canonical_name": p_name, "aliases": []}
    elif p_name not in master["by_key"][p_key]["aliases"] and p_name != master["by_key"][p_key]["canonical_name"]:
        master["by_key"][p_key]["aliases"].append(p_name)
    
    # 2. Registrar Alias
    if norm not in master["by_alias"]:
        master["by_alias"][norm] = p_key
    
    # 3. Alias por inicial (ej. Jesper de Jong -> j. de jong)
    if " " in norm:
        parts = norm.split(" ")
        initial_alias = f"{parts[0][0]}. {' '.join(parts[1:])}"
        if initial_alias not in master["by_alias"]:
            master["by_alias"][initial_alias] = p_key

def calc_h2h_local(p1_key, p2_key, match_date, df_hist):
    """
    Calcula el H2H usando el historial local para evitar llamadas a la API.
    Devuelve (h1_win, h2_win).
    """
    if df_hist.empty or not p1_key or not p2_key:
        return 0, 0
    
    k1, k2 = str(p1_key).replace(".0", ""), str(p2_key).replace(".0", "")
    
    # Filtrar partidos entre ambos ocurridos antes de la fecha actual
    mask = (
        ((df_hist["J1 Key"].astype(str) == k1) & (df_hist["J2 Key"].astype(str) == k2)) |
        ((df_hist["J1 Key"].astype(str) == k2) & (df_hist["J2 Key"].astype(str) == k1))
    ) & (df_hist["Fecha"] < match_date)
    
    matches = df_hist[mask]
    h1_win, h2_win = 0, 0
    
    for _, m in matches.iterrows():
        winner = str(m["Ganador"])
        if winner == "-": continue
        
        # Identificar quién es J1 en este match histórico
        m_j1_name = str(m["Jugador 1"])
        m_j2_name = str(m["Jugador 2"])
        
        # El ganador está guardado por nombre
        if winner == m_j1_name:
            # ¿Es nuestro p1 el J1 de este partido?
            if str(m["J1 Key"]).replace(".0", "") == k1: h1_win += 1
            else: h2_win += 1
        elif winner == m_j2_name:
            if str(m["J2 Key"]).replace(".0", "") == k1: h1_win += 1
            else: h2_win += 1
            
    return h1_win, h2_win


def _get_points(player_name, player_key, rankings_idx, master, target_date):
    """
    Busca puntos usando el índice optimizado.
    """
    if not master or not target_date:
        return ""
    
    if isinstance(target_date, str):
        try:
            target_date = pd.to_datetime(target_date)
        except:
            return ""

    # 1. Por ID directo (más fiable)
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    
    if pk_str and rankings_idx and "by_key" in rankings_idx:
        pts = _get_points_from_index(pk_str, target_date, rankings_idx, use_key=True)
        try: pts_val = int(float(pts))
        except (ValueError, TypeError): pts_val = 0
        if pts_val > 0: return pts_val
    
    # 2. Por nombre (si no hay ID o falló)
    pts = _get_points_from_index(player_name, target_date, rankings_idx)
    try: pts_val = int(float(pts))
    except (ValueError, TypeError): pts_val = 0
    if pts_val > 0: return pts_val
    
    # 3. Fallback: Probar con el nombre canónico del maestro
    if pk_str and pk_str in master["by_key"]:
        canonical = master["by_key"][pk_str]["canonical_name"]
        pts = _get_points_from_index(canonical, target_date, rankings_idx)
        try: pts_val = int(float(pts))
        except (ValueError, TypeError): pts_val = 0
        if pts_val > 0: return pts_val

    return ""


# Variables globales para caché de estadísticas
_SCALE_DATA = None
_RANKINGS_INDEX = None  # Formato: {nombre: [(timestamp, puntos), ...]}

def _load_stats_resources(force_update=False):
    global _SCALE_DATA, _RANKINGS_INDEX
    
    # 1. Cargar Escala (siempre igual)
    if _SCALE_DATA is None:
        try:
            _SCALE_DATA = pd.read_csv("data/Escala ATP - ELO.csv")
        except:
            _SCALE_DATA = pd.DataFrame()

    # 2. Cargar/Actualizar Rankings
    if _RANKINGS_INDEX is None or force_update:
        try:
            df = _get_manager().load_table("rankings")
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Verificar si necesitamos actualización semanal (Auto-completado solicitado)
            last_date = df['date'].max()
            today = datetime.now()
            # Si el último ranking es de hace más de 7 días, o si hoy es Lunes/Martes y el último es de la semana pasada
            # (Los rankings suelen salir los lunes)
            days_since_last = (today - last_date).days
            
            # Lógica: Si han pasado +7 días, intentamos actualizar el archivo
            if days_since_last >= 7:
                logging.info(f"Detectada desincronización de rankings (último: {last_date.date()}). Actualizando...")
                try:
                    from scripts.tenis_api import get_standings
                    new_st = get_standings(event_type='ATP', save_json=False)
                    if new_st is not None and not new_st.empty:
                        # Limpiar y preparar
                        new_st['date'] = pd.to_datetime(new_st['download_time']).dt.normalize()
                        # Incluir player_key si está disponible en la respuesta de la API
                        cols_available = ['player', 'points', 'date']
                        if 'player_key' in new_st.columns:
                            cols_available = ['player', 'player_key', 'points', 'date']
                        new_rows = new_st[cols_available].rename(columns={'player': 'player_name'})
                        # Limpiar player_key: convertir a str entero limpio
                        if 'player_key' in new_rows.columns:
                            new_rows['player_key'] = new_rows['player_key'].apply(
                                lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() not in ['', 'nan'] else ''
                            )
                        # Solo añadir si la fecha es realmente nueva
                        if new_rows['date'].iloc[0] > last_date:
                            df = pd.concat([df, new_rows], ignore_index=True)
                            # Asegurar orden de columnas canónico: player_name, player_key, points, date
                            canon_cols = ['player_name', 'player_key', 'points', 'date']
                            extra_cols = [c for c in df.columns if c not in canon_cols]
                            df = df[[c for c in canon_cols if c in df.columns] + extra_cols]
                            _get_manager().save_table("rankings", df)
                            logging.info(f"Rankings actualizados y guardados. Fecha: {new_rows['date'].iloc[0].date()}")
                except Exception as e:
                    logging.warning(f"No se pudo realizar el auto-completado de rankings: {e}")

            # Construir Índice de búsqueda rápida normalizado y por KEY
            df["norm_name"] = df["player_name"].apply(normalize_name_robust)
            _RANKINGS_INDEX = {"by_key": {}, "by_name": {}}
            
            # Cargar maestro para poder mapear nombres a llaves durante la indexación
            master = load_player_master()
            
            # Determinar si el CSV ya trae player_key directamente
            has_csv_key = "player_key" in df.columns
            
            # Agrupar por nombre normalizado
            for norm_name, group in df.groupby("norm_name"):
                if not norm_name: continue
                sorted_group = group.sort_values("date", ascending=False)
                history = list(zip(sorted_group["date"], sorted_group["points"]))
                
                # 1. Registrar por nombre
                _RANKINGS_INDEX["by_name"][norm_name] = history
                
                # 2. Registrar por KEY — PRIORIDAD: player_key del CSV propio
                p_key = None
                if has_csv_key:
                    # Tomar la primera key válida del grupo
                    for raw_key in sorted_group["player_key"].dropna():
                        clean = str(raw_key).replace(".0", "").strip()
                        if clean and clean not in ("", "nan"):
                            p_key = clean
                            break
                
                # Fallback: buscar key en el maestro de alias
                if not p_key:
                    p_key = master["by_alias"].get(norm_name)
                
                if p_key:
                    if p_key not in _RANKINGS_INDEX["by_key"]:
                        _RANKINGS_INDEX["by_key"][p_key] = history
                    else:
                        # Si ya hay historia para esa key (de otro alias), fusionar y re-ordenar
                        combined = _RANKINGS_INDEX["by_key"][p_key] + history
                        combined.sort(key=lambda x: x[0], reverse=True)
                        _RANKINGS_INDEX["by_key"][p_key] = combined
                
        except Exception as e:
            logging.error(f"Error cargando rankings: {e}")
            _RANKINGS_INDEX = {}
            
    return _SCALE_DATA, _RANKINGS_INDEX

def _get_points_from_index(query, target_date, index, use_key=False):
    """Búsqueda eficiente de puntos en el índice para una fecha dada (por nombre o por key)."""
    sub_index = index["by_key"] if use_key else index["by_name"]
    key = query if use_key else normalize_name_robust(query)
    
    if not key or key not in sub_index:
        return 0
    
    player_history = sub_index[key]
    # player_history está ordenado por fecha desc
    for d, pts in player_history:
        if d <= target_date:
            return pts
    return 0


def calc_performance_refresh(player_name, match_date, surface, df_hist, master, player_key=None, days=365, hist_idx=None):
    """
    Calcula rendimiento usando J1 Key/J2 Key con fallback a búsqueda por alias.
    """
    if hist_idx is not None:
        if pd.isna(match_date) or not master: return "N/D", "N/D"
        if isinstance(match_date, str): match_date = pd.to_datetime(match_date)
        period_start = match_date - pd.Timedelta(days=days)
        pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
        p_name_norm = f"name:{player_name.lower().strip()}"
        
        matches = []
        if pk_str and pk_str in hist_idx:
            matches = hist_idx[pk_str]
        elif p_name_norm in hist_idx:
            matches = hist_idx[p_name_norm]
        else:
            # check aliases
            p_key = master["by_alias"].get(player_name.lower().strip())
            if p_key and p_key in hist_idx: matches = hist_idx[p_key]
            
        if not matches: return "N/D", "N/D"
        
        valid_matches = [m for m in matches if period_start <= m["fecha"] < match_date]
        if not valid_matches: return "N/D", "N/D"
        
        wins = sum(1 for m in valid_matches if m["is_win"])
        recent_pct = f"{(wins / len(valid_matches) * 100):.1f}%"
        
        surf_matches = [m for m in valid_matches if m["superficie"] == surface]
        if not surf_matches:
            surface_pct = "N/D"
        else:
            wins_surf = sum(1 for m in surf_matches if m["is_win"])
            surface_pct = f"{(wins_surf / len(surf_matches) * 100):.1f}%"
            
        return recent_pct, surface_pct

    if df_hist is not None and df_hist.empty or pd.isna(match_date) or not master:
        return "N/D", "N/D"

    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)

    search_names = {player_name.lower().strip()}
    p_key = master["by_alias"].get(player_name.lower().strip())
    if p_key and p_key in master["by_key"]:
        search_names.add(master["by_key"][p_key]["canonical_name"].lower().strip())
        for alias in master["by_key"][p_key]["aliases"]:
            search_names.add(alias.lower().strip())

    period_start = match_date - pd.Timedelta(days=days)
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    
    # Filtrar partidos válidos (Excluyendo pendientes, cancelados, retirados y walkover)
    invalid_states = ["-", "Cancelado", "Retirado", "Walkover", "nan", "None"]
    mask = (df_hist["Fecha"] >= period_start) & (df_hist["Fecha"] < match_date) & (~df_hist["Ganador"].astype(str).isin(invalid_states))
    df_period = df_hist[mask].copy()
    
    if pk_str:
        cond_j1 = (df_period["J1 Key"].astype(str) == pk_str) if "J1 Key" in df_period.columns else pd.Series(False, index=df_period.index)
        cond_j2 = (df_period["J2 Key"].astype(str) == pk_str) if "J2 Key" in df_period.columns else pd.Series(False, index=df_period.index)
        cond = cond_j1 | cond_j2 | \
               (df_period["Jugador 1"].str.lower().str.strip().isin(search_names)) | \
               (df_period["Jugador 2"].str.lower().str.strip().isin(search_names))
    else:
        cond = (df_period["Jugador 1"].str.lower().str.strip().isin(search_names)) | \
               (df_period["Jugador 2"].str.lower().str.strip().isin(search_names))
               
    df_player = df_period[cond]
    total = len(df_player)
    
    if total == 0:
        recent_pct = "N/D"
    else:
        wins = 0
        for _, m in df_player.iterrows():
            if pk_str and (str(m.get("J1 Key", "")) == pk_str or str(m.get("J2 Key", "")) == pk_str):
                # ID Match
                is_j1 = str(m.get("J1 Key", "")) == pk_str
            else:
                # Name Match
                is_j1 = m["Jugador 1"].lower().strip() in search_names
            
            p_name = m["Jugador 1"] if is_j1 else m["Jugador 2"]
            if m["Ganador"] == p_name: wins += 1
            
        recent_pct = f"{(wins / total * 100):.1f}%"
        
    df_surf = df_player[df_player["Superficie"] == surface]
    total_surf = len(df_surf)
    if total_surf == 0:
        surface_pct = "N/D"
    else:
        wins_surf = 0
        for _, m in df_surf.iterrows():
            if pk_str and (str(m.get("J1 Key", "")) == pk_str or str(m.get("J2 Key", "")) == pk_str):
                is_j1 = str(m.get("J1 Key", "")) == pk_str
            else:
                is_j1 = m["Jugador 1"].lower().strip() in search_names
            p_name = m["Jugador 1"] if is_j1 else m["Jugador 2"]
            if m["Ganador"] == p_name: wins_surf += 1
        surface_pct = f"{(wins_surf / total_surf * 100):.1f}%"
        
    return recent_pct, surface_pct


def calc_ultra_performance_refresh(player_name, match_date, df_hist, master, player_key=None, hist_idx=None):
    """
    [v5] Calcula el rendimiento ultra reciente infalible usando Alias nativos (J1 Key/J2 Key).
    """
    if hist_idx is not None:
        if pd.isna(match_date) or not master: return "N/D"
        if isinstance(match_date, str): match_date = pd.to_datetime(match_date)
        date_limit = match_date - pd.Timedelta(days=30)
        
        df_scale, rankings_idx = _load_stats_resources()
        if df_scale.empty or not rankings_idx: return "N/D"
        
        scale_list = list(df_scale.itertuples(index=False, name=None))
        scale_list.sort(key=lambda x: x[0], reverse=True)
        def get_scale_value(points):
            for pts_limit, val in scale_list:
                if points >= pts_limit: return val
            return 40
            
        pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
        p_name_norm = f"name:{player_name.lower().strip()}"
        
        matches = []
        if pk_str and pk_str in hist_idx:
            matches = hist_idx[pk_str]
        elif p_name_norm in hist_idx:
            matches = hist_idx[p_name_norm]
        else:
            p_key = master["by_alias"].get(player_name.lower().strip())
            if p_key and p_key in hist_idx: matches = hist_idx[p_key]
            
        valid_matches = [m for m in matches if date_limit <= m["fecha"] < match_date]
        if not valid_matches: return "N/D"
        
        match_valuations = []
        for m in valid_matches:
            opp_id_str = str(m.get("opp_key", "")).replace(".0","")
            opp_id = int(float(opp_id_str)) if opp_id_str and opp_id_str != "nan" else -1
            opp_name = m.get("opp_name", "")
            
            opp_points = 0
            if opp_id != -1: opp_points = _get_points_from_index(str(opp_id), m["fecha"], rankings_idx, use_key=True)
            if opp_points == 0 and opp_name: opp_points = _get_points_from_index(opp_name, m["fecha"], rankings_idx)
            
            scale_val = get_scale_value(opp_points)
            sign = 1 if m["is_win"] else -1
            base_score = sign * scale_val
            
            days_diff = (match_date - m["fecha"]).days
            multiplier = 1.0 - (max(0, days_diff - 1) / 100.0)
            match_valuations.append(base_score * multiplier)
            
        if not match_valuations: return "N/D"
        avg_score = sum(match_valuations) / len(match_valuations)
        fpct = (50.0 + 0.5 * avg_score) / 100.0
        return f"{(max(0.0, min(1.0, fpct)) * 100):.1f}%"

    if df_hist is not None and df_hist.empty or pd.isna(match_date) or not master:
        return "N/D"

    p_norm = player_name.lower().strip()
    search_names = {p_norm}
    p_key_dict = master["by_alias"].get(p_norm)
    p_key = int(float(p_key_dict)) if p_key_dict else -1
    canonical_p = player_name
    
    if p_key != -1 and str(p_key) in master["by_key"]:
        canonical_p = master["by_key"][str(p_key)]["canonical_name"]
        search_names.add(canonical_p.lower().strip())
        for alias in master["by_key"][str(p_key)]["aliases"]:
            search_names.add(alias.lower().strip())

    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)
    
    df_scale, rankings_idx = _load_stats_resources()
    if df_scale.empty or not rankings_idx:
        return "N/D"
        
    date_limit = match_date - pd.Timedelta(days=30)
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    
    if pk_str:
        cond_j1 = (df_hist["J1 Key"].astype(str) == pk_str) if "J1 Key" in df_hist.columns else pd.Series(False, index=df_hist.index)
        cond_j2 = (df_hist["J2 Key"].astype(str) == pk_str) if "J2 Key" in df_hist.columns else pd.Series(False, index=df_hist.index)
        cond = cond_j1 | cond_j2 | \
               (df_hist["Jugador 1"].str.lower().str.strip().isin(search_names)) | \
               (df_hist["Jugador 2"].str.lower().str.strip().isin(search_names))
    else:
        cond = (df_hist["Jugador 1"].str.lower().str.strip().isin(search_names)) | \
               (df_hist["Jugador 2"].str.lower().str.strip().isin(search_names))
               
    # Filtrar partidos válidos de los últimos 30 días
    invalid_states = ["-", "Cancelado", "Retirado", "Walkover", "nan", "None"]
    # Corregir filtro para excluir estados inválidos
    df_player = df_hist[cond & (df_hist["Fecha"] >= date_limit) & (df_hist["Fecha"] < match_date) & (~df_hist["Ganador"].astype(str).isin(invalid_states))]
    
    if df_player.empty:
        return "N/D"
        
    match_valuations = []
    scale_list = list(df_scale.itertuples(index=False, name=None))
    scale_list.sort(key=lambda x: x[0], reverse=True)

    def get_scale_value(points):
        for pts_limit, val in scale_list:
            if points >= pts_limit:
                return val
        return 40

    for _, match in df_player.iterrows():
        if pk_str and (str(match.get("J1 Key", "")) == pk_str or str(match.get("J2 Key", "")) == pk_str):
            is_j1 = str(match.get("J1 Key", "")) == pk_str
        else:
            is_j1 = match["Jugador 1"].lower().strip() in search_names
            
        m_j1, m_j2 = match["Jugador 1"], match["Jugador 2"]
        opponent_name = m_j2 if is_j1 else m_j1
        my_name = m_j1 if is_j1 else m_j2
        is_win = match["Ganador"] == my_name
        
        m_dt = pd.to_datetime(match["Fecha"])
        
        # Oponente identity via ID
        opp_id_str = str(match.get("J2 Key" if is_j1 else "J1 Key", "")).replace(".0","")
        opp_id = int(float(opp_id_str)) if opp_id_str and opp_id_str != "nan" else -1
        if opp_id == -1:
            o_key_dict = master["by_alias"].get(opponent_name.lower().strip())
            opp_id = int(float(o_key_dict)) if o_key_dict else -1
            
        opp_canonical = opponent_name
        if opp_id != -1 and str(opp_id) in master["by_key"]:
            opp_canonical = master["by_key"][str(opp_id)]["canonical_name"]

        # ── Búsqueda de puntos del rival: PRIORIDAD por ID ────────────────
        # Paso 1: búsqueda directa por player_key (O(1), sin ambigüedad de nombres)
        opp_points = 0
        if opp_id != -1:
            opp_points = _get_points_from_index(str(opp_id), m_dt, rankings_idx, use_key=True)

        # Paso 2: fallback por nombre canónico
        if opp_points == 0:
            opp_points = _get_points_from_index(opp_canonical, m_dt, rankings_idx)

        # Paso 3: fallback por nombre original (si el canónico difiere)
        if opp_points == 0 and opp_canonical != opponent_name:
            opp_points = _get_points_from_index(opponent_name, m_dt, rankings_idx)

        # Paso 4: si sigue sin puntos, el rival es jugador de muy bajo ranking →
        # get_scale_value(0) ya devuelve 40 (mínimo de la escala), pero lo forzamos
        # explícitamente para que quede claro en el flujo.
        # (No se requiere acción: get_scale_value maneja el caso con el return 40 final)


        scale_val = get_scale_value(opp_points)
        sign = 1 if is_win else -1
        base_score = sign * scale_val
        
        days_diff = (match_date - m_dt).days
        multiplier = 1.0 - (max(0, days_diff - 1) / 100.0)
        
        match_valuations.append(base_score * multiplier)
        
    if not match_valuations:
        return "N/D"
        
    avg_score = sum(match_valuations) / len(match_valuations)
    fpct = (50.0 + 0.5 * avg_score) / 100.0
    fpct = max(0.0, min(1.0, fpct))
    return f"{(fpct * 100):.1f}%"


def refresh(mode="ATP", progress_callback=None) -> dict:
    """
    Actualiza de forma incremental:
    - Mode 'ATP': Histórico + data/ATP Tour 2026 Matches.csv
    - Mode 'CHA': Histórico + data/Challenger Tour Matches.csv

    progress_callback(step, total, msg) se llama para informar el progreso.
    Devuelve un dict con resumen del resultado.
    """
    if mode == "CHA":
        TARGET_FILE = CHA26_FILE
        TARGET_TYPES = CHA_TYPES
        TARGET_LABEL = "Challenger 2026"
    else:
        TARGET_FILE = ATP26_FILE
        TARGET_TYPES = ATP_TYPES
        TARGET_LABEL = "ATP 2026"

    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    future_limit = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
    smap = _surface_map()
    summary = {}
    all_unknowns = []

    # ── 1. HISTÓRICO (ATP + Challenger) ──────────────────────────
    df_hist = _get_manager().load_table("historical_fixtures")
    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], errors="coerce")
    hist_max = df_hist["Fecha"].max()

    def _build_hist_index(df_h):
        idx = {}
        invalid = {"-", "Cancelado", "Retirado", "Walkover", "nan", "None"}
        df_valid = df_h[~df_h["Ganador"].astype(str).isin(invalid)]
        for _, row in df_valid.iterrows():
            dt = row["Fecha"]
            if pd.isna(dt): continue
            surf = row.get("Superficie", "")
            win = str(row.get("Ganador", ""))
            
            j1k = str(row.get("J1 Key", "")).replace(".0", "").strip()
            j1n = str(row.get("Jugador 1", "")).lower().strip()
            j2k = str(row.get("J2 Key", "")).replace(".0", "").strip()
            j2n = str(row.get("Jugador 2", "")).lower().strip()
            
            m_j1 = {"fecha": dt, "superficie": surf, "is_win": win == str(row.get("Jugador 1", "")), "opp_name": row.get("Jugador 2", ""), "opp_key": j2k}
            m_j2 = {"fecha": dt, "superficie": surf, "is_win": win == str(row.get("Jugador 2", "")), "opp_name": row.get("Jugador 1", ""), "opp_key": j1k}
            
            if j1k and j1k not in ("nan", "None"): idx.setdefault(j1k, []).append(m_j1)
            else: idx.setdefault(f"name:{j1n}", []).append(m_j1)
            if j2k and j2k not in ("nan", "None"): idx.setdefault(j2k, []).append(m_j2)
            else: idx.setdefault(f"name:{j2n}", []).append(m_j2)
        return idx
        
    hist_idx = _build_hist_index(df_hist)

    # Retroceder 3 días para asegurar que completamos el último día si hubo cambios o fallos (según feedback)
    hist_from = (hist_max - timedelta(days=3)).strftime("%Y-%m-%d")

    if hist_from <= today:
        if progress_callback:
            progress_callback(1, 4, f"Histórico: descargando desde {hist_from} hasta {today}…")

        # Descarga por bloques de 7 días
        cursor = datetime.strptime(hist_from, "%Y-%m-%d")
        end    = datetime.strptime(today,  "%Y-%m-%d")
        new_hist = []

        while cursor <= end:
            block_end = min(cursor + timedelta(days=6), end)
            chunk = fetch_range(cursor.strftime("%Y-%m-%d"), block_end.strftime("%Y-%m-%d"))
            filtered = [r for r in chunk if r.get("event_type_type") in HIST_TYPES]
            for f in filtered:
                surf = _get_surface(f, smap)
                if surf == "Unknown":
                    t_name = f.get("tournament_name", "Desconocido")
                    if not any(u["name"] == t_name for u in all_unknowns):
                        all_unknowns.append({"name": t_name, "key": f.get("tournament_key")})
                    # Aunque sea unknown, recolectamos el match para procesarlo después si se define
                
                new_hist.append({
                    "Fecha":     f.get("event_date"),
                    "Hora":      f.get("event_time", ""),
                    "Torneo":    f.get("tournament_name"),
                    "Superficie": surf,
                    "Jugador 1":  f.get("event_first_player", ""),
                    "J1 Key":     str(f.get("first_player_key", "")) if f.get("first_player_key") else "",
                    "Jugador 2":  f.get("event_second_player", ""),
                    "J2 Key":     str(f.get("second_player_key", "")) if f.get("second_player_key") else "",
                    "Ganador":    _winner(f),
                    "ID Partido": str(f.get("event_key", "")),
                })
            cursor = block_end + timedelta(days=1)
            time.sleep(0.4)

        if new_hist:
            df_new = pd.DataFrame(new_hist)
            df_new["Fecha"] = pd.to_datetime(df_new["Fecha"], errors="coerce")
            df_new = _normalize_surface(df_new)
            
            # Sincronización y guardado se difieren hasta verificar que no hay unknown_surfaces
            summary["hist_added"] = len(new_hist)
        else:
            summary["hist_added"] = 0
    else:
        summary["hist_added"] = 0
        if progress_callback:
            progress_callback(1, 4, "Histórico: ya está al día.")

    # ── 2. ACTUALIZACIÓN ARCHIVO ESPECÍFICO (ATP o Challenger) ─────
    table_name = "atp_matches" if mode == "ATP" else "challenger_matches"
    df_target = _get_manager().load_table(table_name)
    # Obtener rankings actuales para rellenar puntos
    if progress_callback:
        progress_callback(2, 4, "Obteniendo rankings ATP actuales…")
    # ── 2. PREPARACIÓN DE RECURSOS (Rankings + Master) ────────────
    try:
        _, rankings_idx = _load_stats_resources()
        master = load_player_master()
    except Exception as e:
        logging.error(f"Error cargando recursos en refresh: {e}")
        rankings_idx = {}
        master = {"by_key": {}, "by_alias": {}}

    # Convertir fechas (formato m/d/yy que usa ese archivo)
    df_target["_fecha_dt"] = pd.to_datetime(df_target["Fecha"], format="%m/%d/%y", errors="coerce")
    df_target = df_target.drop(columns=["_fecha_dt"])

    # VENTANA FIJA: Siempre descargamos los últimos 7 días hasta el futuro.
    # Esto garantiza que aunque el archivo tenga datos futuros, siempre se revisan
    # los días recientes y no se pierde ningún partido nuevo de la API.
    target_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    if target_from <= future_limit:
        # Optimización: Cargar IDs ya procesados con estadísticas
        existing_ids_with_stats = set()
        if not df_target.empty and "ID Partido" in df_target.columns:
            # Consideramos procesado si tiene ID y alguna estadística clave (ej. J1 Rend. Reciente)
            # Manejamos NaN y strings vacíos
            mask_done = df_target["ID Partido"].notna() & (df_target["ID Partido"].astype(str).str.strip() != "") & (df_target["ID Partido"].astype(str).str.lower() != "nan")
            
            if "J1 Rend. Reciente" in df_target.columns:
                mask_done &= df_target["J1 Rend. Reciente"].notna() & (df_target["J1 Rend. Reciente"].astype(str).str.strip() != "") & (df_target["J1 Rend. Reciente"].fillna("N/D") != "N/D")
            
            existing_ids_with_stats = set()
            for val in df_target[mask_done]["ID Partido"].astype(str):
                # Eliminar '.0' si pandas lo convirtió a float string
                clean_id = val.replace(".0", "").strip()
                if clean_id:
                    existing_ids_with_stats.add(clean_id)

        if progress_callback:
            progress_callback(3, 4, f"{TARGET_LABEL}: descargando desde {target_from} hasta {future_limit}…")

        cursor = datetime.strptime(target_from, "%Y-%m-%d")
        end    = datetime.strptime(future_limit, "%Y-%m-%d")
        new_matches = []

        while cursor <= end:
            block_end = min(cursor + timedelta(days=6), end)
            start_str = cursor.strftime("%Y-%m-%d")
            end_str = block_end.strftime("%Y-%m-%d")
            
            if progress_callback:
                progress_callback(2, 4, f"Descargando {TARGET_LABEL} API ({start_str} a {end_str})...")
            
            chunk = fetch_range(start_str, end_str)
            odds_map = get_odds_data(start_str, end_str)
            
            filtered = [r for r in chunk if r.get("event_type_type") in TARGET_TYPES]

            for f in filtered:
                match_key = str(f.get("event_key", ""))
                
                # SKIP LOGIC: Usar el set de IDs para velocidad extrema (O(1))
                if match_key in existing_ids_with_stats:
                    logging.debug(f"Saltando partido ya procesado: {match_key}")
                    continue

                surf = _get_surface(f, smap)
                if surf == "Unknown":
                    t_name = f.get("tournament_name", "Desconocido")
                    if not any(u["name"] == t_name for u in all_unknowns):
                        all_unknowns.append({"name": t_name, "key": f.get("tournament_key")})
                    continue # No procesar hasta que se defina la superficie

                # Cuotas: Priorizar el mapa específico de cuotas
                if match_key in odds_map:
                    odd1, odd2 = odds_map[match_key]
                else:
                    try:
                        odd1 = float(f.get("event_odd_1") or 0)
                    except Exception:
                        odd1 = 0.0
                    try:
                        odd2 = float(f.get("event_odd_2") or 0)
                    except Exception:
                        odd2 = 0.0

                surf = _get_surface(f, smap)
                if "Hard" in surf or surf == "Unknown":
                    surf = "Hard"
                if "Clay" in surf:
                    surf = "Clay"
                if "Grass" in surf:
                    surf = "Grass"

                # Fecha en formato m/d/yy (igual al CSV)
                raw_date = f.get("event_date", "")
                try:
                    raw_date_dt = datetime.strptime(raw_date, "%Y-%m-%d")
                    fecha_fmt = raw_date_dt.strftime("%-m/%-d/%y")
                except Exception:
                    raw_date_dt = today_dt # fallback
                    fecha_fmt = raw_date

                # Obtener H2H
                h1_win, h1_pct, h2_win, h2_pct = 0, "0%", 0, "0%"
                p1_key = f.get("first_player_key")
                p2_key = f.get("second_player_key")
                
                if p1_key and p2_key:
                    # PRIORIDAD: Calcular H2H localmente (Instantáneo)
                    h1_win, h2_win = calc_h2h_local(p1_key, p2_key, raw_date_dt, df_hist)
                    
                    # FALLBACK: Si no hay partidos en el historial, consultar API
                    if h1_win == 0 and h2_win == 0:
                        h2h_res = get_h2h(p1_key, p2_key, save_json=False, save_csv=False)
                        if h2h_res and "H2H" in h2h_res:
                            matches = h2h_res["H2H"]
                            for m in matches:
                                h2h_date = m.get("event_date", "")
                                if h2h_date and raw_date and str(h2h_date) < str(raw_date):
                                    winner = m.get("event_winner")
                                    if winner == "First Player": h1_win += 1
                                    elif winner == "Second Player": h2_win += 1
                            # Pequeña espera SOLO si llamamos a la API
                            time.sleep(0.5)
                        
                    total_h2h = h1_win + h2_win
                    if total_h2h > 0:
                        h1_pct = f"{(h1_win / total_h2h * 100):.0f}%"
                        h2_pct = f"{(h2_win / total_h2h * 100):.0f}%"

                # Obtener Rendimiento (basado en df_hist que acabamos de actualizar)
                r1, s1 = calc_performance_refresh(f.get("event_first_player"), raw_date, surf, None, master, player_key=p1_key, hist_idx=hist_idx)
                r2, s2 = calc_performance_refresh(f.get("event_second_player"), raw_date, surf, None, master, player_key=p2_key, hist_idx=hist_idx)
                u1 = calc_ultra_performance_refresh(f.get("event_first_player"), raw_date, None, master, player_key=p1_key, hist_idx=hist_idx)
                u2 = calc_ultra_performance_refresh(f.get("event_second_player"), raw_date, None, master, player_key=p2_key, hist_idx=hist_idx)

                # Actualizar Maestro de Jugadores (Aprendizaje)
                update_player_master(master, p1_key, f.get("event_first_player"))
                update_player_master(master, p2_key, f.get("event_second_player"))

                new_matches.append({
                    "Torneo":     f.get("tournament_name", ""),
                    "Fecha":      fecha_fmt,
                    "Hora":       f.get("event_time", ""),
                    "Superficie": surf,
                    "Jugador 1":  f.get("event_first_player", ""),
                    "J1 Key":     str(p1_key) if p1_key else "",
                    "J1 Puntos ATP": _get_points(f.get("event_first_player"), p1_key, rankings_idx, master, raw_date_dt),
                    "Jugador 2":  f.get("event_second_player", ""),
                    "J2 Key":     str(p2_key) if p2_key else "",
                    "J2 Puntos ATP": _get_points(f.get("event_second_player"), p2_key, rankings_idx, master, raw_date_dt),
                    "J1 H2H":    h1_win if h1_win > 0 else 0,
                    "J1 H2H %":  h1_pct,
                    "J2 H2H":    h2_win if h2_win > 0 else 0,
                    "J2 H2H %":  h2_pct,
                    "J1 Rend. Reciente": r1,
                    "J1 Rend. Superficie": s1,
                    "Rend. Ultra reciente J1": u1,
                    "J2 Rend. Reciente": r2,
                    "J2 Rend. Superficie": s2,
                    "Rend. Ultra reciente J2": u2,
                    "Cuota J1":  odd1 if odd1 else "",
                    "Cuota J2":  odd2 if odd2 else "",
                    "Ganador":   _winner(f),
                    "ID Partido": match_key,
                })
                # Pequeña espera para no saturar la API con H2H
                time.sleep(0.5)

            cursor = block_end + timedelta(days=1)
            time.sleep(0.4)

        if all_unknowns:
            return {"status": "NEED_SURFACE", "tournaments": all_unknowns, "until": yesterday}

        # --- FINALIZACIÓN HISTÓRICO (Solo si no hay unknown_surfaces) ---
        if 'df_new' in locals():
            # Unir y limpiar Histórico
            df_hist = pd.concat([df_hist, df_new], ignore_index=True)
            if "ID Partido" in df_hist.columns:
                df_hist["_dedup_id"] = df_hist["ID Partido"].fillna("").astype(str).str.replace(".0", "", regex=False).str.strip()
                mask_no_id = (df_hist["_dedup_id"] == "") | (df_hist["_dedup_id"].str.lower() == "nan")
                df_hist.loc[mask_no_id, "_dedup_id"] = (
                    df_hist.loc[mask_no_id, "Fecha"].dt.strftime("%Y-%m-%d") + "_" +
                    df_hist.loc[mask_no_id, "Torneo"].astype(str) + "_" +
                    df_hist.loc[mask_no_id, "Jugador 1"].astype(str) + "_" +
                    df_hist.loc[mask_no_id, "Jugador 2"].astype(str)
                )
                df_hist = df_hist.drop_duplicates(subset=["_dedup_id"], keep="last").drop(columns=["_dedup_id"])
            else:
                df_hist = df_hist.drop_duplicates(subset=["Fecha", "Torneo", "Jugador 1", "Jugador 2"], keep="last")
            df_hist = df_hist.sort_values("Fecha", ascending=False)
            _get_manager().save_table("historical_fixtures", df_hist)


        if 'new_matches' in locals() and new_matches:
            df_new_matches = pd.DataFrame(new_matches)
            df_new_matches["_dt_temp"] = pd.to_datetime(df_new_matches["Fecha"], format="%m/%d/%y", errors="coerce")

            # ── Deduplicación con ventana ±3 días (por J1 Key + J2 Key) ──────────────────
            # Construir índice: (j1k, j2k) -> [(row_idx, fecha)] de los partidos existentes
            df_target["_dt_temp"] = pd.to_datetime(df_target["Fecha"], format="%m/%d/%y", errors="coerce")
            df_target["_j1k_idx"] = df_target["J1 Key"].astype(str).str.replace(".0", "", regex=False)
            df_target["_j2k_idx"] = df_target["J2 Key"].astype(str).str.replace(".0", "", regex=False)
            key_date_index = {}
            for idx_ar, ar in df_target.iterrows():
                for k in [(ar["_j1k_idx"], ar["_j2k_idx"]), (ar["_j2k_idx"], ar["_j1k_idx"])]:
                    if k not in key_date_index:
                        key_date_index[k] = []
                    if pd.notna(ar["_dt_temp"]):
                        key_date_index[k].append((idx_ar, ar["_dt_temp"]))
            df_target = df_target.drop(columns=["_dt_temp", "_j1k_idx", "_j2k_idx"])

            def _find_existing_nearby(j1k, j2k, match_dt, window=3):
                """Devuelve (row_idx, existing_dt) del partido cercano o (None, None) si no existe."""
                for k in [(j1k, j2k), (j2k, j1k)]:
                    for (row_idx, existing_dt) in key_date_index.get(k, []):
                        if abs((match_dt - existing_dt).days) <= window:
                            return row_idx, existing_dt
                return None, None

            truly_new = []
            date_updates = 0
            for row_dict in new_matches:
                j1k = str(row_dict.get("J1 Key", "")).replace(".0", "")
                j2k = str(row_dict.get("J2 Key", "")).replace(".0", "")
                row_dt_str = row_dict.get("Fecha", "")
                try:
                    row_dt = pd.to_datetime(row_dt_str, format="%m/%d/%y", errors="coerce")
                    if pd.isna(row_dt):
                        row_dt = pd.to_datetime(row_dt_str, errors="coerce")
                except Exception:
                    row_dt = pd.NaT

                existing_idx, existing_dt = _find_existing_nearby(j1k, j2k, row_dt) if pd.notna(row_dt) else (None, None)

                if existing_idx is None:
                    # Partido realmente nuevo: añadir bloque completo
                    truly_new.append(row_dict)
                else:
                    # Duplicado: Aplicar Regla Técnica Definitiva (Sincronización 7 días)
                    now = datetime.now()
                    days_diff = (now - existing_dt).days
                    
                    if days_diff > 7:
                        # REGLA: > 7 días, omitir completamente (Inviolabilidad Histórica)
                        continue
                    
                    # REGLA: <= 7 días, permitir actualización PARCIAL de Ganador y Cuotas si están vacíos
                    match_key = row_dict.get("ID Partido")
                    
                    # 1. Asignar ID si falta
                    existing_id = str(df_target.at[existing_idx, "ID Partido"]).strip()
                    if existing_id == "" or existing_id.lower() == "nan":
                        df_target.at[existing_idx, "ID Partido"] = match_key

                    # 2. Actualizar Ganador si está pendiente ("-")
                    existing_win = str(df_target.at[existing_idx, "Ganador"]).strip()
                    new_win = str(row_dict.get("Ganador", "-")).strip()
                    if existing_win == "-" and new_win != "-":
                        df_target.at[existing_idx, "Ganador"] = new_win
                        logging.info(f"Ganador actualizado (delta 7d): {row_dict.get('Jugador 1')} vs {row_dict.get('Jugador 2')} ({row_dt.date()}) -> {new_win}")

                    # 3. Actualizar Puntos ATP si faltan
                    for p_col, p_name_col, p_key_col in [("J1 Puntos ATP", "Jugador 1", "J1 Key"), ("J2 Puntos ATP", "Jugador 2", "J2 Key")]:
                        curr_pts = df_target.at[existing_idx, p_col]
                        # Considerar vacío si es nan, "" o "N/D"
                        if pd.isna(curr_pts) or str(curr_pts).strip() in ["", "nan", "N/D"]:
                            p_name = df_target.at[existing_idx, p_name_col]
                            p_key = df_target.at[existing_idx, p_key_col]
                            p_key_clean = str(p_key).replace(".0", "") if pd.notna(p_key) else None
                            new_pts = _get_points(p_name, p_key_clean, rankings_idx, master, existing_dt)
                            if new_pts:
                                df_target.at[existing_idx, p_col] = new_pts
                                logging.info(f"Puntos ATP actualizados (delta 7d) para {p_name}: {new_pts}")

                    # 4. Actualizar Cuotas si faltan
                    for c_key in ["Cuota J1", "Cuota J2"]:
                        curr_val = df_target.at[existing_idx, c_key]
                        if pd.isna(curr_val) or str(curr_val).strip() == "":
                            df_target.at[existing_idx, c_key] = row_dict.get(c_key)

                    # 4. Actualizar fecha si es más reciente
                    if pd.notna(row_dt) and row_dt > existing_dt:
                        df_target.at[existing_idx, "Fecha"] = row_dt_str
                        date_updates += 1

            if truly_new:
                df_truly_new = pd.DataFrame(truly_new)
                df_target = pd.concat([df_target, df_truly_new], ignore_index=True)
                logging.info(f"API download: {len(truly_new)} partidos nuevos añadidos, {date_updates} fechas actualizadas (de {len(new_matches)} descargados).")
            elif date_updates > 0:
                logging.info(f"API download: 0 partidos nuevos, {date_updates} fechas actualizadas.")
            # ──────────────────────────────────────────────────────────────
            save_player_master(master)
            summary["target_added"] = len(truly_new)
            _get_manager().save_table(table_name, df_target)
        else:
            summary["target_added"] = 0
    else:
        summary["target_added"] = 0
        if progress_callback:
            progress_callback(3, 4, f"{TARGET_LABEL}: ya está al día.")

    # ── 2.5. SINCRONIZACIÓN CON HISTÓRICO (Fuente de Verdad) ─────────────────────────────
    # El histórico contiene TODOS los partidos ATP+Challenger pasados.
    # Lo usamos para: (a) detectar partidos ATP ATP que faltan en ATP26, y
    # (b) actualizar el campo Ganador de partidos pendientes sin llamar a la API.
    if progress_callback:
        progress_callback(3, 4, "Sincronizando con histórico como fuente de verdad…")

    try:
        df_hist_sync = _get_manager().load_table("historical_fixtures")
        df_hist_sync["Fecha"] = pd.to_datetime(df_hist_sync["Fecha"], errors="coerce")

        # Construir el conjunto de torneos ATP que ya conocemos en el archivo ATP26
        target_known_torneos = set(df_target["Torneo"].dropna().unique())

        # Filtrar histórico: solo partidos de 2026 en torneos que ya aparecen en ATP26
        # (esto excluye automáticamente el otro circuito, ITF, etc.)
        df_hist_target = df_hist_sync[
            (df_hist_sync["Fecha"].dt.year == 2026) &
            (df_hist_sync["Torneo"].isin(target_known_torneos))
        ].copy()

        if not df_hist_target.empty:
            # ── (a) Detectar partidos faltantes en ATP26 con ventana ±3 días ─────────
            df_target["_j1k"] = df_target["J1 Key"].astype(str).str.replace(".0", "", regex=False)
            df_target["_j2k"] = df_target["J2 Key"].astype(str).str.replace(".0", "", regex=False)
            df_hist_target["_j1k"] = df_hist_target["J1 Key"].astype(str).str.replace(".0", "", regex=False)
            df_hist_target["_j2k"] = df_hist_target["J2 Key"].astype(str).str.replace(".0", "", regex=False)

            # Construir índice (j1k, j2k) -> [(row_idx, fecha)] del archivo ATP26 actual
            df_target["_dt_atp"] = pd.to_datetime(df_target["Fecha"], format="%m/%d/%y", errors="coerce")
            key_date_idx_sync = {}
            for idx_ar, ar in df_target.iterrows():
                for k in [(ar["_j1k"], ar["_j2k"]), (ar["_j2k"], ar["_j1k"])]:
                    if k not in key_date_idx_sync:
                        key_date_idx_sync[k] = []
                    if pd.notna(ar["_dt_atp"]):
                        key_date_idx_sync[k].append((idx_ar, ar["_dt_atp"]))
            df_target = df_target.drop(columns=["_dt_atp"])

            def _hist_match_exists(j1k, j2k, match_date, window=3):
                """True si ya existe un partido con las mismas claves a ±window días en el archivo destino."""
                if not pd.notna(match_date):
                    return False
                for k in [(j1k, j2k), (j2k, j1k)]:
                    for (_, existing_dt) in key_date_idx_sync.get(k, []):
                        if abs((match_date - existing_dt).days) <= window:
                            return True
                return False

            # Detectar faltantes; para los duplicados, actualizar fecha si la del hist es más reciente
            sync_date_updates = 0
            missing_rows = []
            for _, h_row in df_hist_target.iterrows():
                j1k = h_row["_j1k"]
                j2k = h_row["_j2k"]
                match_date = h_row["Fecha"]

                found_idx = None
                found_dt = None
                if pd.notna(match_date):
                    for k in [(j1k, j2k), (j2k, j1k)]:
                        for (row_idx, existing_dt) in key_date_idx_sync.get(k, []):
                            if abs((match_date - existing_dt).days) <= 3:
                                found_idx = row_idx
                                found_dt = existing_dt
                                break
                        if found_idx is not None:
                            break

                if found_idx is None:
                    # REGLA: Solo añadir si es RECIENTE (< 7 días)
                    now = datetime.now()
                    days_diff = (now - match_date).days
                    if days_diff <= 7:
                        missing_rows.append(h_row)
                else:
                    # Existe: Aplicar Regla 7 días para sincronización desde histórico
                    now = datetime.now()
                    days_diff = (now - found_dt).days
                    
                    if days_diff <= 7:
                        # 1. Asignar ID si el partido existente no lo tiene
                        if "ID Partido" in df_target.columns and "ID Partido" in h_row:
                            existing_id = str(df_target.at[found_idx, "ID Partido"]).strip()
                            hist_id = str(h_row.get("ID Partido", "")).strip()
                            if (existing_id == "" or existing_id.lower() == "nan") and (hist_id != "" and hist_id.lower() != "nan"):
                                df_target.at[found_idx, "ID Partido"] = hist_id

                        # 2. Actualizar fecha si la del histórico es más reciente
                        if match_date > found_dt:
                            try:
                                new_fecha_fmt = match_date.strftime("%-m/%-d/%y")
                            except Exception:
                                new_fecha_fmt = match_date.strftime("%m/%d/%y")
                            df_target.at[found_idx, "Fecha"] = new_fecha_fmt
                            sync_date_updates += 1
                            logging.info(f"Sync hist: fecha actualizada {h_row.get('Jugador 1')} vs {h_row.get('Jugador 2')}")

            missing_in_target = pd.DataFrame([r for r in missing_rows]) if missing_rows else pd.DataFrame()

            logging.info(f"Sync histórico: {len(missing_in_target)} partidos en hist no encontrados en el archivo destino.")

            new_from_hist = []
            for _, h_row in missing_in_target.iterrows():
                p1_name = h_row["Jugador 1"]
                p2_name = h_row["Jugador 2"]
                p1_key  = h_row["_j1k"]
                p2_key  = h_row["_j2k"]
                match_date = h_row["Fecha"]
                surf = h_row.get("Superficie", "Hard")

                raw_date_str = match_date.strftime("%Y-%m-%d") if pd.notna(match_date) else ""

                try:
                    fecha_fmt = match_date.strftime("%-m/%-d/%y") if pd.notna(match_date) else ""
                except Exception:
                    fecha_fmt = raw_date_str

                # Calcular rendimientos usando el histórico actualizado
                p1k_clean = None if p1_key in ["", "nan", "None"] else p1_key
                p2k_clean = None if p2_key in ["", "nan", "None"] else p2_key

                r1, s1 = calc_performance_refresh(p1_name, raw_date_str, surf, None, master, player_key=p1k_clean, hist_idx=hist_idx)
                r2, s2 = calc_performance_refresh(p2_name, raw_date_str, surf, None, master, player_key=p2k_clean, hist_idx=hist_idx)
                u1 = calc_ultra_performance_refresh(p1_name, raw_date_str, None, master, player_key=p1k_clean, hist_idx=hist_idx)
                u2 = calc_ultra_performance_refresh(p2_name, raw_date_str, None, master, player_key=p2k_clean, hist_idx=hist_idx)

                new_from_hist.append({
                    "Torneo":     h_row["Torneo"],
                    "Fecha":      fecha_fmt,
                    "Superficie": surf,
                    "Jugador 1":  p1_name,
                    "J1 Key":     p1_key,
                    "J1 Puntos ATP": _get_points(p1_name, p1k_clean, rankings_idx, master, raw_date_str),
                    "Jugador 2":  p2_name,
                    "J2 Key":     p2_key,
                    "J2 Puntos ATP": _get_points(p2_name, p2k_clean, rankings_idx, master, raw_date_str),
                    "J1 H2H": 0, "J1 H2H %": "0%",
                    "J2 H2H": 0, "J2 H2H %": "0%",
                    "J2 Rend. Reciente": r2, "J2 Rend. Superficie": s2, "Rend. Ultra reciente J2": u2,
                    "Cuota J1": "", "Cuota J2": "",
                    "Ganador": h_row.get("Ganador", "-"),
                    "ID Partido": h_row.get("ID Partido", ""),
                })

            if new_from_hist:
                df_from_hist = pd.DataFrame(new_from_hist)
                df_target = pd.concat([df_target, df_from_hist], ignore_index=True)
                summary["target_added"] = summary.get("target_added", 0) + len(new_from_hist)
                logging.info(f"Sync histórico: añadidos {len(new_from_hist)} partidos faltantes al archivo destino.")

            # Limpiar columnas temporales de claves
            df_target = df_target.drop(columns=["_j1k", "_j2k"], errors="ignore")

            # ── (b) Actualizar Ganador desde el histórico (vía ID Partido) ─────────────────
            # Solo sincronizamos si coinciden los IDs únicos de partido para evitar cruces
            # entre distintos torneos del mismo enfrentamiento.
            hist_id_winner_map = {}
            for _, h_row in df_hist_sync.iterrows():
                h_id = str(h_row.get("ID Partido", "")).replace(".0", "").strip()
                ganador = h_row.get("Ganador", "-")
                if h_id and h_id.lower() != "nan" and ganador != "-":
                    hist_id_winner_map[h_id] = ganador

            updates_from_hist = 0
            for idx_row, row in df_target[df_target["Ganador"] == "-"].iterrows():
                match_id = str(row.get("ID Partido", "")).replace(".0", "").strip()
                if match_id in hist_id_winner_map:
                    df_target.at[idx_row, "Ganador"] = hist_id_winner_map[match_id]
                    updates_from_hist += 1

            logging.info(f"Sync histórico: {updates_from_hist} ganadores actualizados desde histórico.")

    except Exception as e:
        logging.error(f"Error en sincronización con histórico: {e}")

    # Ordenar por fecha final
    df_target["_sort"] = pd.to_datetime(df_target["Fecha"], format="%m/%d/%y", errors="coerce")
    df_target = df_target.sort_values("_sort").drop(columns=["_sort"])
    
    if progress_callback:
        progress_callback(4, 4, "Finalizando y guardando archivos...")
        
    _get_manager().save_table(table_name, df_target)

    summary["until"] = yesterday
    summary["status"] = "SUCCESS"
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    def log_progress(step, total, msg):
        logging.info(f"[{step}/{total}] {msg}")

    result = refresh(progress_callback=log_progress)
    print(f"\n✅ Refresh completado hasta {result['until']}")
    print(f"   Histórico: +{result['hist_added']} registros")
    print(f"   Target:  +{result.get('target_added', 0)} filas")
