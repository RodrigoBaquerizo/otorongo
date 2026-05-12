import re

with open('scripts/refresh_data.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update gen_download block (Líneas 430-437)
old_append = '''                new_hist.append({
                    "Fecha":     f.get("event_date"),
                    "Torneo":    f.get("tournament_name"),
                    "Superficie": _get_surface(f, smap),
                    "Jugador 1": f.get("event_first_player"),
                    "Jugador 2": f.get("event_second_player"),
                    "Ganador":   _winner(f),
                })'''

new_append = '''                new_hist.append({
                    "Fecha":     f.get("event_date"),
                    "Torneo":    f.get("tournament_name"),
                    "Superficie": _get_surface(f, smap),
                    "Jugador 1":  f.get("event_first_player", ""),
                    "J1 Key":     str(f.get("first_player_key", "")) if f.get("first_player_key") else "",
                    "Jugador 2":  f.get("event_second_player", ""),
                    "J2 Key":     str(f.get("second_player_key", "")) if f.get("second_player_key") else "",
                    "Ganador":    _winner(f),
                })'''
text = text.replace(old_append, new_append)

# 2. Update calc_performance_refresh
# It starts at `def calc_performance_refresh` and ends before `def calc_ultra_performance_refresh`
old_perf = re.search(r'def calc_performance_refresh.*?return recent_pct, surface_pct\s*', text, re.DOTALL).group(0)

new_perf = '''def calc_performance_refresh(player_name, match_date, surface, df_hist, master, player_key=None, months=12):
    """
    Calcula rendimiento usando J1 Key/J2 Key con fallback a búsqueda por alias.
    """
    if df_hist.empty or pd.isna(match_date) or not master:
        return "N/D", "N/D"

    if isinstance(match_date, str):
        match_date = pd.to_datetime(match_date)

    search_names = {player_name.lower().strip()}
    p_key = master["by_alias"].get(player_name.lower().strip())
    if p_key and p_key in master["by_key"]:
        search_names.add(master["by_key"][p_key]["canonical_name"].lower().strip())
        for alias in master["by_key"][p_key]["aliases"]:
            search_names.add(alias.lower().strip())

    period_start = match_date - pd.DateOffset(months=months)
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    
    mask = (df_hist["Fecha"] >= period_start) & (df_hist["Fecha"] < match_date) & (df_hist["Ganador"] != "-")
    df_period = df_hist[mask].copy()
    
    if pk_str:
        cond = (df_period.get("J1 Key", "").astype(str) == pk_str) | \\
               (df_period.get("J2 Key", "").astype(str) == pk_str) | \\
               (df_period["Jugador 1"].str.lower().str.strip().isin(search_names)) | \\
               (df_period["Jugador 2"].str.lower().str.strip().isin(search_names))
    else:
        cond = (df_period["Jugador 1"].str.lower().str.strip().isin(search_names)) | \\
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
\n\n'''
text = text.replace(old_perf, new_perf)

# 3. Update calc_ultra_performance_refresh
old_ultra = re.search(r'def calc_ultra_performance_refresh.*?return f"{\(final_pct \* 100\):\.1f}%"\s*', text, re.DOTALL).group(0)

new_ultra = '''def calc_ultra_performance_refresh(player_name, match_date, df_hist, master, player_key=None):
    """
    [v5] Calcula el rendimiento ultra reciente infalible usando Alias nativos (J1 Key/J2 Key).
    """
    if df_hist.empty or pd.isna(match_date) or not master:
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
    
    df_scale, df_rankings = _load_stats_resources()
    if df_scale.empty or df_rankings.empty:
        return "N/D"
        
    date_limit = match_date - pd.Timedelta(days=30)
    pk_str = str(player_key).replace(".0", "") if player_key and player_key not in ["", "nan", "None"] else None
    
    if pk_str:
        cond = (df_hist.get("J1 Key", "").astype(str) == pk_str) | \\
               (df_hist.get("J2 Key", "").astype(str) == pk_str) | \\
               (df_hist["Jugador 1"].str.lower().str.strip().isin(search_names)) | \\
               (df_hist["Jugador 2"].str.lower().str.strip().isin(search_names))
    else:
        cond = (df_hist["Jugador 1"].str.lower().str.strip().isin(search_names)) | \\
               (df_hist["Jugador 2"].str.lower().str.strip().isin(search_names))
               
    df_player = df_hist[cond & (df_hist["Fecha"] >= date_limit) & (df_hist["Fecha"] < match_date) & (df_hist["Ganador"] != "-")]
    
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
        
        m_date = pd.to_datetime(match["Fecha"])
        
        # Oponente identity via ID
        opp_id_str = str(match.get("J2 Key" if is_j1 else "J1 Key", "")).replace(".0","")
        opp_id = int(float(opp_id_str)) if opp_id_str and opp_id_str != "nan" else -1
        if opp_id == -1:
            o_key_dict = master["by_alias"].get(opponent_name.lower().strip())
            opp_id = int(float(o_key_dict)) if o_key_dict else -1
            
        opp_canonical = opponent_name
        if opp_id != -1 and str(opp_id) in master["by_key"]:
            opp_canonical = master["by_key"][str(opp_id)]["canonical_name"]

        opp_points = 0
        if opp_id != -1 and "player_id" in df_rankings.columns:
            df_opp_r = df_rankings[df_rankings["player_id"] == opp_id]
            df_opp_r_b = df_opp_r[df_opp_r["date"] <= m_date]
            if not df_opp_r_b.empty:
                opp_points = df_opp_r_b.sort_values("date", ascending=False).iloc[0]["points"]
        
        if opp_points == 0:
            df_opp_r_n = df_rankings[df_rankings["player_name"] == opp_canonical]
            df_opp_r_b_n = df_opp_r_n[df_opp_r_n["date"] <= m_date]
            if not df_opp_r_b_n.empty:
                opp_points = df_opp_r_b_n.sort_values("date", ascending=False).iloc[0]["points"]

        if opp_points == 0 and opp_canonical != opponent_name:
            df_opp_r_n = df_rankings[df_rankings["player_name"] == opponent_name]
            df_opp_r_b_n = df_opp_r_n[df_opp_r_n["date"] <= m_date]
            if not df_opp_r_b_n.empty:
                opp_points = df_opp_r_b_n.sort_values("date", ascending=False).iloc[0]["points"]
            
        scale_val = get_scale_value(opp_points)
        sign = 1 if is_win else -1
        base_score = sign * scale_val
        
        days_diff = (match_date - m_date).days
        multiplier = 1.0 - (max(0, days_diff - 1) / 100.0)
        
        match_valuations.append(base_score * multiplier)
        
    if not match_valuations:
        return "N/D"
        
    avg_score = sum(match_valuations) / len(match_valuations)
    fpct = (50.0 + 0.5 * avg_score) / 100.0
    fpct = max(0.0, min(1.0, fpct))
    return f"{(fpct * 100):.1f}%"
\n\n'''
text = text.replace(old_ultra, new_ultra)

# 4. Actualizar llamadas a las funciones en la creacion de nuevo partido
text = text.replace('r1, s1 = calc_performance_refresh(f.get("event_first_player"), raw_date, surf, df_hist, master)',
                    'r1, s1 = calc_performance_refresh(f.get("event_first_player"), raw_date, surf, df_hist, master, player_key=p1_key)')
text = text.replace('r2, s2 = calc_performance_refresh(f.get("event_second_player"), raw_date, surf, df_hist, master)',
                    'r2, s2 = calc_performance_refresh(f.get("event_second_player"), raw_date, surf, df_hist, master, player_key=p2_key)')
text = text.replace('u1 = calc_ultra_performance_refresh(f.get("event_first_player"), raw_date, df_hist, master)',
                    'u1 = calc_ultra_performance_refresh(f.get("event_first_player"), raw_date, df_hist, master, player_key=p1_key)')
text = text.replace('u2 = calc_ultra_performance_refresh(f.get("event_second_player"), raw_date, df_hist, master)',
                    'u2 = calc_ultra_performance_refresh(f.get("event_second_player"), raw_date, df_hist, master, player_key=p2_key)')

# 5. Actualizar llamadas en backfill
text = text.replace('r1, s1 = calc_performance_refresh(p1, dt_match, surf, df_hist, master)',
                    'r1, s1 = calc_performance_refresh(p1, dt_match, surf, df_hist, master, player_key=p1_key_clean)')
text = text.replace('r2, s2 = calc_performance_refresh(p2, dt_match, surf, df_hist, master)',
                    'r2, s2 = calc_performance_refresh(p2, dt_match, surf, df_hist, master, player_key=p2_key_clean)')
text = text.replace('u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master)',
                    'u1 = calc_ultra_performance_refresh(p1, dt_match, df_hist, master, player_key=p1_key_clean)')
text = text.replace('u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master)',
                    'u2 = calc_ultra_performance_refresh(p2, dt_match, df_hist, master, player_key=p2_key_clean)')

with open('scripts/refresh_data.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated refresh_data.py successfully!")
