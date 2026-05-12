import re

with open("scripts/refresh_data.py", "r") as f:
    code = f.read()

# 1. Update refresh() variable names (df_atp -> df_target, new_atp -> new_matches)
# df_atp is already mostly replaced by df_target in some places, but let's check.
code = code.replace("df_atp", "df_target")
code = code.replace("new_atp", "new_matches")
code = code.replace("df_hist_atp", "df_hist_target")

# 2. Fix hardcoded ATP26 logic in refresh()
# Line ~1000: "# Filtrar histórico: solo partidos de 2026 en torneos que ya aparecen en ATP26"
code = code.replace(
    '# (esto excluye automáticamente Challengers, ITF, etc.)\n        df_hist_target = df_hist_sync[\n            (df_hist_sync["Fecha"].dt.year == 2026) &\n            (df_hist_sync["Torneo"].isin(target_known_torneos))\n        ].copy()',
    '# (esto excluye automáticamente el otro circuito, ITF, etc.)\n        df_hist_target = df_hist_sync[\n            (df_hist_sync["Fecha"].dt.year == 2026) &\n            (df_hist_sync["Torneo"].isin(target_known_torneos))\n        ].copy()'
)

# Replace 'en ATP26' with 'en el target'
code = code.replace('en ATP26.', 'en el archivo destino.')
code = code.replace('a ATP26.', 'al archivo destino.')

# 3. Add `hist_idx=None` to function signatures
code = code.replace(
    'def calc_performance_refresh(player_name, match_date, surface, df_hist, master, player_key=None, days=365):',
    'def calc_performance_refresh(player_name, match_date, surface, df_hist, master, player_key=None, days=365, hist_idx=None):'
)

code = code.replace(
    'def calc_ultra_performance_refresh(player_name, match_date, df_hist, master, player_key=None):',
    'def calc_ultra_performance_refresh(player_name, match_date, df_hist, master, player_key=None, hist_idx=None):'
)

# 4. Modify calc_performance_refresh
# We will inject the fast path at the beginning
fast_path_perf = """    if hist_idx is not None:
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
"""
code = code.replace(
    '    if df_hist.empty or pd.isna(match_date) or not master:\n        return "N/D", "N/D"',
    fast_path_perf + '\n    if df_hist is not None and df_hist.empty or pd.isna(match_date) or not master:\n        return "N/D", "N/D"'
)

# 5. Modify calc_ultra_performance_refresh
fast_path_ultra = """    if hist_idx is not None:
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
"""

code = code.replace(
    '    if df_hist.empty or pd.isna(match_date) or not master:\n        return "N/D"',
    fast_path_ultra + '\n    if df_hist is not None and df_hist.empty or pd.isna(match_date) or not master:\n        return "N/D"'
)

# 6. Inject build_hist_index and hist_idx usage into refresh()
build_idx_code = """
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
"""

code = code.replace(
    '    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], errors="coerce")\n    hist_max = df_hist["Fecha"].max()',
    '    df_hist["Fecha"] = pd.to_datetime(df_hist["Fecha"], errors="coerce")\n    hist_max = df_hist["Fecha"].max()\n' + build_idx_code
)

# 7. Update calc calls to pass hist_idx in refresh()
# Around line 813 (fetching from API)
code = code.replace(
    'r1, s1 = calc_performance_refresh(f.get("event_first_player"), raw_date, surf, df_hist, master, player_key=p1_key)',
    'r1, s1 = calc_performance_refresh(f.get("event_first_player"), raw_date, surf, None, master, player_key=p1_key, hist_idx=hist_idx)'
)
code = code.replace(
    'r2, s2 = calc_performance_refresh(f.get("event_second_player"), raw_date, surf, df_hist, master, player_key=p2_key)',
    'r2, s2 = calc_performance_refresh(f.get("event_second_player"), raw_date, surf, None, master, player_key=p2_key, hist_idx=hist_idx)'
)
code = code.replace(
    'u1 = calc_ultra_performance_refresh(f.get("event_first_player"), raw_date, df_hist, master, player_key=p1_key)',
    'u1 = calc_ultra_performance_refresh(f.get("event_first_player"), raw_date, None, master, player_key=p1_key, hist_idx=hist_idx)'
)
code = code.replace(
    'u2 = calc_ultra_performance_refresh(f.get("event_second_player"), raw_date, df_hist, master, player_key=p2_key)',
    'u2 = calc_ultra_performance_refresh(f.get("event_second_player"), raw_date, None, master, player_key=p2_key, hist_idx=hist_idx)'
)

# Around line 1108 (Syncing from History)
code = code.replace(
    'r1, s1 = calc_performance_refresh(p1_name, raw_date_str, surf, df_hist, master, player_key=p1k_clean)',
    'r1, s1 = calc_performance_refresh(p1_name, raw_date_str, surf, None, master, player_key=p1k_clean, hist_idx=hist_idx)'
)
code = code.replace(
    'r2, s2 = calc_performance_refresh(p2_name, raw_date_str, surf, df_hist, master, player_key=p2k_clean)',
    'r2, s2 = calc_performance_refresh(p2_name, raw_date_str, surf, None, master, player_key=p2k_clean, hist_idx=hist_idx)'
)
code = code.replace(
    'u1 = calc_ultra_performance_refresh(p1_name, raw_date_str, df_hist, master, player_key=p1k_clean)',
    'u1 = calc_ultra_performance_refresh(p1_name, raw_date_str, None, master, player_key=p1k_clean, hist_idx=hist_idx)'
)
code = code.replace(
    'u2 = calc_ultra_performance_refresh(p2_name, raw_date_str, df_hist, master, player_key=p2k_clean)',
    'u2 = calc_ultra_performance_refresh(p2_name, raw_date_str, None, master, player_key=p2k_clean, hist_idx=hist_idx)'
)

with open("scripts/refresh_data.py", "w") as f:
    f.write(code)

print("Refactor applied.")
