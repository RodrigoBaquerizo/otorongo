import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import logging
import time
from datetime import datetime, timedelta
import base64
from scripts.data_manager import DataManager

@st.cache_resource
def get_data_manager():
    return DataManager()

manager = get_data_manager()

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configurar página
st.set_page_config(page_title="🎾 Otorongo - Tennis Analytics", layout="wide", page_icon="🐆")

# Inicialización de Estado Pendiente para Refresh Auto-Resume
if "pending_refresh_mode" not in st.session_state:
    st.session_state.pending_refresh_mode = None

# --- CUSTOM CSS ---
def load_css():
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# --- UTILS ---
def parse_pct(val):
    if pd.isna(val) or val in ["N/D", "-", "", "nan", "nan%"]:
        return 0.0
    try:
        if isinstance(val, str):
            return float(val.replace("%", "").replace(",", ".").strip())
        return float(val)
    except:
        return 0.0

def norm(val):
    if pd.isna(val) or val is None: return ""
    return str(val).lower().replace(".", "").replace(" ", "").strip()

# --- AUTH ---
def check_password():
    if "password_correct" in st.session_state and st.session_state["password_correct"]:
        return True

    password = os.getenv("APP_PASSWORD")
    if not password:
        return True # Sin protección si no hay ENV

    def password_entered():
        if st.session_state["password_input"] == password:
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    st.title("🎾 Acceso Reservado")
    st.text_input("Contraseña", type="password", on_change=password_entered, key="password_input")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Contraseña incorrecta")
    return False

if not check_password():
    st.stop()

# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_tournaments_data_v2():
    try:
        return manager.load_table("tournaments")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_standings_data(event_type="ATP"):
    try:
        return manager.load_table("rankings")
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_matches_data(table_name):
    try:
        df = manager.load_table(table_name)
        if not df.empty:
            if "Fecha" in df.columns:
                df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
            
            # Coercionar columnas estrictamente numéricas (TEXT) a float
            numeric_cols = ["Cuota J1", "Cuota J2", "J1 Puntos ATP", "J2 Puntos ATP"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        logging.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()

def load_analysis_config_v2(config_file):
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                cfg = json.load(f)
            # One-time migration: if new surface keys don't exist, copy from legacy keys
            if "td_hard_weight_h2h" not in cfg:
                surf_keys = ["weight_h2h", "weight_recent", "weight_surface", "weight_ranking", "weight_ultra", "min_prob", "min_prob_no_h2h"]
                for k in surf_keys:
                    legacy_val = cfg.get(f"td_{k}", None)
                    if legacy_val is not None:
                        cfg[f"td_hard_{k}"] = legacy_val
                        cfg[f"td_clay_{k}"] = legacy_val
                # Remove legacy generic keys
                for k in surf_keys:
                    cfg.pop(f"td_{k}", None)
                try:
                    os.makedirs("data", exist_ok=True)
                    with open(config_file, "w") as f:
                        json.dump(cfg, f, indent=4)
                except Exception:
                    pass
            return cfg
        except:
            pass
    return {}

def save_analysis_config_v2(config_file, prefix):
    # Global keys (shared across surfaces)
    global_keys = ["max_bet", "min_bet", "min_odds", "min_atp", "visible_cols"]
    # Per-surface keys
    surf_keys = ["weight_h2h", "weight_recent", "weight_surface", "weight_ranking", "weight_ultra", "min_prob", "min_prob_no_h2h"]
    config = {}
    for k in global_keys:
        sk = f"{prefix}_{k}"
        if sk in st.session_state:
            config[f"td_{k}"] = st.session_state[sk]
    for surf in ["hard", "clay"]:
        for k in surf_keys:
            sk = f"{prefix}_{surf}_{k}"
            if sk in st.session_state:
                config[f"td_{surf}_{k}"] = st.session_state[sk]
    try:
        os.makedirs("data", exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        st.toast("✅ Configuración guardada")
    except Exception as e:
        st.error(f"Error al guardar configuración: {e}")

# --- CALCULATION ENGINE ---
def calculate_match_stats(row):
    """
    Simulación de la lógica de negocio estable.
    Esta función es invocada por el diálogo de detalles.
    """
    stats = {
        "Date": row.get("Fecha", "Unknown"),
        "Player 1": row.get("Jugador 1", "N/D"),
        "Player 2": row.get("Jugador 2", "N/D"),
        "Sourface": row.get("Superficie", "Unknown"),
        "H2H P1": row.get("J1 H2H", 0),
        "H2H % P1": row.get("J1 H2H %", "0%"),
        "P1 Rec. Performance": row.get("J1 Rend. Reciente", "N/D"),
        "P1 Sourface R. Perf.": row.get("J1 Rend. Superficie", "N/D"),
        "P1 Ultra Rec. Performance": row.get("Rend. Ultra reciente J1", "N/D"),
        "P1 ATP Points": row.get("J1 Puntos ATP", 0),
        "P2 Rec. Performance": row.get("J2 Rend. Reciente", "N/D"),
        "P2 Sourface R. Perf.": row.get("J2 Rend. Superficie", "N/D"),
        "P2 Ultra Rec. Performance": row.get("Rend. Ultra reciente J2", "N/D"),
        "P2 ATP Points": row.get("J2 Puntos ATP", 0),
    }
    return stats

# --- UI COMPONENTS ---
@st.dialog("Detalles del Partido")
def show_details_dialog(row):
    st.markdown("""
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 85vw;
            max-width: 85vw;
        }
        </style>
    """, unsafe_allow_html=True)
    
    stats = calculate_match_stats(row)
    st.subheader(f"🐆 Análisis: {stats['Player 1']} vs {stats['Player 2']}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Torneo:** {row.get('Torneo', 'N/D')}")
        st.write(f"**Superficie:** {stats['Sourface']}")
    with col2:
        st.write(f"**Fecha:** {stats['Date']}")
        st.write(f"**ID Partido:** {row.get('ID Partido', 'N/D')}")
    
    st.divider()
    
    # Tabla comparativa
    display_keys = ["H2H", "Recent %", "Surface %", "Ultra %", "ATP Points"]
    p1_vals = [stats["H2H % P1"], stats["P1 Rec. Performance"], stats["P1 Sourface R. Perf."], stats["P1 Ultra Rec. Performance"], stats["P1 ATP Points"]]
    p2_h2h_pct = row.get("J2 H2H %", "0%")
    p2_vals = [p2_h2h_pct, stats["P2 Rec. Performance"], stats["P2 Sourface R. Perf."], stats["P2 Ultra Rec. Performance"], stats["P2 ATP Points"]]
    
    df_cmp = pd.DataFrame({
        "Métrica": display_keys,
        stats["Player 1"]: p1_vals,
        stats["Player 2"]: p2_vals
    })
    st.table(df_cmp)
    
    st.info("Nota: Los detalles históricos completos están disponibles en la tabla principal de datos.")

# --- DIALOGOS ---
@st.dialog("Asignar Superficies a Torneos Nuevos")
def show_surface_assignment_dialog(tournaments, mode):
    st.warning("Se han detectado nuevos torneos en la API sin superficie asignada.")
    st.write("Por favor, asigna la superficie correspondiente para continuar con la actualización.")
    
    with st.form("surface_assignment_form"):
        selections = {}
        for t in tournaments:
            key = t.get("key", "N/D")
            name = t.get("name", "Desconocido")
            selections[key] = {
                "name": name,
                "surface": st.selectbox(f"{name} (Key: {key})", ["Hard", "Clay", "Grass"], key=f"sel_{key}")
            }
            
        if st.form_submit_button("Confirmar y Guardar", type="primary"):
            try:
                # 1. Cargar torneos existentes mediante el Manager
                df_trn = manager.load_table("tournaments")
                
                # 2. Preparar nuevos registros
                new_entries = []
                for k, data in selections.items():
                    new_entries.append({
                        "tournament_key": str(k),
                        "tournament_name": data["name"],
                        "tournament_sourface": data["surface"]
                    })
                
                df_new = pd.DataFrame(new_entries)
                
                # 3. Concatenar y guardar usando persistencia dual del Manager
                if df_trn.empty:
                    df_final = df_new
                else:
                    df_final = pd.concat([df_trn, df_new], ignore_index=True)
                
                # Deduplicar por key por si acaso
                df_final = df_final.drop_duplicates(subset=["tournament_key"], keep="last")
                
                success = manager.save_table("tournaments", df_final)
                
                if success:
                    st.success("✅ Superficies guardadas correctamente. Reanudando actualización...")
                    # El refresco de la app continuará automáticamente gracias a st.session_state.pending_refresh_mode
                    st.rerun()
                else:
                    st.error("❌ Fallo crítico al guardar las superficies en la base de datos.")
            except Exception as e:
                st.error(f"❌ Error al guardar: {e}")

# --- CORE CALCULATION ENGINE ---

def _compute_bets_df(df_raw, prefix, date_start, date_end, surf_filter):
    """
    Aplica el motor de cálculo completo (pesos, umbrales, doble gatillo)
    sobre df_raw y devuelve el DataFrame enriquecido con Bet?, Won?, PNL, Amount.
    Lee los parámetros directamente desde st.session_state usando el prefijo dado.
    """
    df = df_raw.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df[(df["Fecha"].dt.date >= date_start) & (df["Fecha"].dt.date <= date_end)]
    if surf_filter != "Todas":
        df = df[df["Superficie"] == surf_filter]
    if df.empty:
        return df

    # Load both surface parameter sets
    def _ss(key): return st.session_state[key]

    hard_w_h2h  = _ss(f"{prefix}_hard_weight_h2h") / 100
    hard_w_rec  = _ss(f"{prefix}_hard_weight_recent") / 100
    hard_w_sur  = _ss(f"{prefix}_hard_weight_surface") / 100
    hard_w_ran  = _ss(f"{prefix}_hard_weight_ranking") / 100
    hard_w_ult  = _ss(f"{prefix}_hard_weight_ultra") / 100
    hard_m_prob        = _ss(f"{prefix}_hard_min_prob")
    hard_m_prob_no_h2h = _ss(f"{prefix}_hard_min_prob_no_h2h")

    clay_w_h2h  = _ss(f"{prefix}_clay_weight_h2h") / 100
    clay_w_rec  = _ss(f"{prefix}_clay_weight_recent") / 100
    clay_w_sur  = _ss(f"{prefix}_clay_weight_surface") / 100
    clay_w_ran  = _ss(f"{prefix}_clay_weight_ranking") / 100
    clay_w_ult  = _ss(f"{prefix}_clay_weight_ultra") / 100
    clay_m_prob        = _ss(f"{prefix}_clay_min_prob")
    clay_m_prob_no_h2h = _ss(f"{prefix}_clay_min_prob_no_h2h")

    # Global constants
    m_odds = _ss(f"{prefix}_min_odds")
    m_atp  = _ss(f"{prefix}_min_atp")
    max_b  = _ss(f"{prefix}_max_bet")
    min_b  = _ss(f"{prefix}_min_bet")

    # Select weights per row: Clay → clay params, everything else → hard params (fallback)
    is_clay = (df["Superficie"].str.lower() == "clay").values
    w_h2h = np.where(is_clay, clay_w_h2h, hard_w_h2h)
    w_rec  = np.where(is_clay, clay_w_rec,  hard_w_rec)
    w_sur  = np.where(is_clay, clay_w_sur,  hard_w_sur)
    w_ran  = np.where(is_clay, clay_w_ran,  hard_w_ran)
    w_ult  = np.where(is_clay, clay_w_ult,  hard_w_ult)
    m_prob_arr        = np.where(is_clay, clay_m_prob,        hard_m_prob)
    m_prob_no_h2h_arr = np.where(is_clay, clay_m_prob_no_h2h, hard_m_prob_no_h2h)

    def get_f_norm(v1, v2, weight):
        denom = v1 + v2
        return np.where(denom > 0, (v1 / denom) * weight, 0)

    h1 = df["J1 H2H %"].apply(parse_pct)
    h2 = df["J2 H2H %"].apply(parse_pct)
    r1 = df["J1 Rend. Reciente"].apply(parse_pct)
    r2 = df["J2 Rend. Reciente"].apply(parse_pct)
    s1 = df["J1 Rend. Superficie"].apply(parse_pct)
    s2 = df["J2 Rend. Superficie"].apply(parse_pct)
    u1 = df["Rend. Ultra reciente J1"].apply(parse_pct)
    u2 = df["Rend. Ultra reciente J2"].apply(parse_pct)
    p1 = pd.to_numeric(df["J1 Puntos ATP"], errors="coerce").fillna(0)
    p2 = pd.to_numeric(df["J2 Puntos ATP"], errors="coerce").fillna(0)
    o1 = pd.to_numeric(df["Cuota J1"], errors="coerce").fillna(0)
    o2 = pd.to_numeric(df["Cuota J2"], errors="coerce").fillna(0)

    df["f1"] = ((h1/100 * w_h2h) + get_f_norm(r1, r2, w_rec) + get_f_norm(s1, s2, w_sur) + get_f_norm(p1, p2, w_ran) + get_f_norm(u1, u2, w_ult)) * 100.0
    df["f2"] = ((h2/100 * w_h2h) + get_f_norm(r2, r1, w_rec) + get_f_norm(s2, s1, w_sur) + get_f_norm(p2, p1, w_ran) + get_f_norm(u2, u1, w_ult)) * 100.0

    def get_stake(prob_pct, odds, p1_pts, p2_pts, has_h2h, thr, thr_no_h2h):
        threshold = thr if has_h2h else thr_no_h2h
        if prob_pct < threshold: return 0, "No"
        if odds < m_odds: return 0, "No"
        if (p1_pts + p2_pts) < m_atp: return 0, "No"
        p = prob_pct / 100.0
        if p >= 0.62: pct = 1.0
        elif p >= 0.59: pct = 0.60
        elif p >= 0.56: pct = 0.36
        elif p >= 0.53: pct = 0.264
        elif p >= 0.50: pct = 0.20
        else: pct = 0.0
        amt = max_b * pct
        if amt > 0 and amt < min_b: amt = min_b
        return amt, "Yes"

    has_h2h_series = (h1 > 0) | (h2 > 0)
    stakes_1 = [get_stake(f, o, pv1, pv2, hh, thr, thr_nh) for f, o, pv1, pv2, hh, thr, thr_nh in zip(df["f1"], o1, p1, p2, has_h2h_series, m_prob_arr, m_prob_no_h2h_arr)]
    stakes_2 = [get_stake(f, o, pv1, pv2, hh, thr, thr_nh) for f, o, pv1, pv2, hh, thr, thr_nh in zip(df["f2"], o2, p1, p2, has_h2h_series, m_prob_arr, m_prob_no_h2h_arr)]

    bet_flags, bet_for, amounts = [], [], []
    for i in range(len(df)):
        s1_amt, s1_flag = stakes_1[i]
        s2_amt, s2_flag = stakes_2[i]
        if s1_flag == "Yes" and s2_flag == "Yes":
            if df.iloc[i]["f1"] >= df.iloc[i]["f2"]:
                s2_flag, s2_amt = "No", 0
            else:
                s1_flag, s1_amt = "No", 0
        if s1_flag == "Yes":
            bet_flags.append("Yes"); bet_for.append(df.iloc[i]["Jugador 1"]); amounts.append(s1_amt)
        elif s2_flag == "Yes":
            bet_flags.append("Yes"); bet_for.append(df.iloc[i]["Jugador 2"]); amounts.append(s2_amt)
        else:
            bet_flags.append("No"); bet_for.append(""); amounts.append(0)

    df["Bet?"]   = bet_flags
    df["Bet for"] = bet_for
    df["Amount"] = amounts

    def check_win(row):
        if row["Bet?"] == "No": return ""
        gn = norm(row["Ganador"])
        if not gn or gn == "-": return ""
        if any(e in gn for e in ["retired","cancelled","walkover","retirado","cancelado"]): return ""
        return "Yes" if norm(row["Bet for"]) == gn else "No"

    df["Won?"] = df.apply(check_win, axis=1)

    def calc_pnl(row):
        if row["Won?"] == "": return 0.0
        if row["Won?"] == "Yes":
            cuota = float(row["Cuota J1"] if norm(row["Bet for"]) == norm(row["Jugador 1"]) else row["Cuota J2"])
            return row["Amount"] * (cuota - 1)
        return -row["Amount"]

    df["PNL"] = df.apply(calc_pnl, axis=1)
    return df


# --- TABS RENDERERS ---

def render_stats_analysis_tab(header, table_name, config_path, prefix, mode, button_label):
    st.header(header)
    
    # --- Configuración Persistence ---
    saved_cfg = load_analysis_config_v2(config_path)
    
    # Global constants (shared across surfaces)
    global_defaults = {"max_bet": 100.0, "min_bet": 20.0, "min_odds": 1.08, "min_atp": 0}
    for k, default in global_defaults.items():
        sk = f"{prefix}_{k}"
        if sk not in st.session_state:
            st.session_state[sk] = saved_cfg.get(f"td_{k}", default)

    # Per-surface weights and thresholds
    surf_defaults = {
        "weight_h2h": 1.0, "weight_recent": 15.0, "weight_surface": 43.8,
        "weight_ranking": 40.5, "weight_ultra": 0.0,
        "min_prob": 64.0, "min_prob_no_h2h": 60.0
    }
    for surf in ["hard", "clay"]:
        for k, default in surf_defaults.items():
            sk = f"{prefix}_{surf}_{k}"
            if sk not in st.session_state:
                st.session_state[sk] = saved_cfg.get(f"td_{surf}_{k}", default)

    # --- Acciones ---
    col_ref, _ = st.columns([2, 8])
    with col_ref:
        if st.button(button_label, key=f"btn_{prefix}_refresh", type="primary", use_container_width=True):
            st.session_state.pending_refresh_mode = mode
            st.rerun()

    # --- Cargar Datos ---
    df_raw = load_matches_data(table_name)
    
    if df_raw.empty:
        st.warning(f"No se encontraron datos en la tabla {table_name}")
        return
        
    # --- Ordenamiento Predeterminado por Fecha ---
    df_raw["Fecha"] = pd.to_datetime(df_raw["Fecha"], errors="coerce")
    df_raw = df_raw.sort_values(by="Fecha", ascending=False).reset_index(drop=True)

    # --- Filtros Temporales ---
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        date_start = st.date_input("Fecha Inicio", value=df_raw["Fecha"].min().date(), key=f"{prefix}_date_start")
    with col_f2:
        date_end = st.date_input("Fecha Fin", value=df_raw["Fecha"].max().date(), key=f"{prefix}_date_end")
    with col_f3:
        surf_filter = st.selectbox("Superficie", ["Todas", "Hard", "Clay", "Grass"], key=f"{prefix}_surf_filter")

    st.divider()

    # --- Constantes Globales (compartidas) ---
    st.markdown("##### ⚙️ Constantes Globales")
    gc1, gc2, gc3, gc4 = st.columns(4)
    gc1.number_input("Apuesta máx. (€)", value=float(st.session_state[f"{prefix}_max_bet"]), step=10.0, key=f"{prefix}_max_bet")
    gc2.number_input("Apuesta mín. (€)", value=float(st.session_state[f"{prefix}_min_bet"]), step=5.0, key=f"{prefix}_min_bet")
    gc3.number_input("Cuota mínima", value=float(st.session_state[f"{prefix}_min_odds"]), step=0.01, key=f"{prefix}_min_odds")
    gc4.number_input("Puntos ATP mín.", value=int(st.session_state[f"{prefix}_min_atp"]), step=10, key=f"{prefix}_min_atp")

    st.markdown("##### 🏟️ Configuración por Superficie")
    col_hard, col_clay = st.columns(2)

    for surf, col, emoji in [("hard", col_hard, "🏟️ Dura (Hard)"), ("clay", col_clay, "🌿 Tierra (Clay)")]:
        with col:
            st.markdown(f"**{emoji}**")
            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.number_input("H2H (%)", value=float(st.session_state[f"{prefix}_{surf}_weight_h2h"]), step=0.5, key=f"{prefix}_{surf}_weight_h2h")
            r1c2.number_input("Reciente (%)", value=float(st.session_state[f"{prefix}_{surf}_weight_recent"]), step=0.5, key=f"{prefix}_{surf}_weight_recent")
            r1c3.number_input("Superficie (%)", value=float(st.session_state[f"{prefix}_{surf}_weight_surface"]), step=0.5, key=f"{prefix}_{surf}_weight_surface")
            
            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.number_input("Ranking (%)", value=float(st.session_state[f"{prefix}_{surf}_weight_ranking"]), step=0.5, key=f"{prefix}_{surf}_weight_ranking")
            r2c2.number_input("Ultra (%)", value=float(st.session_state[f"{prefix}_{surf}_weight_ultra"]), step=0.5, key=f"{prefix}_{surf}_weight_ultra")
            r2c3.number_input("Prob. Mín. (%)", value=float(st.session_state[f"{prefix}_{surf}_min_prob"]), step=0.5, key=f"{prefix}_{surf}_min_prob")
            
            r3c1, r3c2, r3c3 = st.columns(3)
            r3c1.number_input("Prob. Mín. (No H2H)", value=float(st.session_state[f"{prefix}_{surf}_min_prob_no_h2h"]), step=0.5, key=f"{prefix}_{surf}_min_prob_no_h2h")

    if st.button("💾 Guardar Configuración", key=f"btn_save_{prefix}", use_container_width=False):
        save_analysis_config_v2(config_path, prefix)

    # --- Filtrado y Cálculos (usando motor centralizado) ---
    df = _compute_bets_df(df_raw, prefix, date_start, date_end, surf_filter)

    if df.empty:
        st.info("No hay partidos que coincidan con los filtros de fecha/superficie.")
        return

    # Persistir el df calculado en session_state para que las pestañas de análisis puedan consumirlo
    if prefix == "atp":
        st.session_state["atp_computed_df"] = df
    elif prefix == "cha":
        st.session_state["cha_computed_df"] = df


    # --- TABLA DE RESUMEN EJECUTIVO (ESTABLE) ---
    finished_bets = df[df["Won?"].isin(["Yes", "No"])]
    wins = (finished_bets["Won?"] == "Yes").sum()
    losses = (finished_bets["Won?"] == "No").sum()
    
    total_matches = len(df)
    total_bets_fin = len(finished_bets)
    win_rate = (wins / total_bets_fin * 100) if total_bets_fin > 0 else 0
    total_pnl = df["PNL"].sum()
    total_invested = finished_bets["Amount"].sum()
    roi = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    df_resumen = pd.DataFrame({
        "Partidos": [total_matches],
        "Apostados (Fin.)": [total_bets_fin],
        "Acierto": [f"{win_rate:.2f}% ({wins}W-{losses}L)"],
        "Balance": [f"{total_pnl:.2f} €"],
        "ROI": [f"{roi:.2f}%"]
    })
    
    def color_financials(val):
        try:
            num = float(str(val).replace(' €', '').replace('%', ''))
            if num > 0: return 'color: #28a745; font-weight: bold;'
            elif num < 0: return 'color: #dc3545; font-weight: bold;'
            return ''
        except:
            return ''
            
    styled_resumen = df_resumen.style.map(color_financials, subset=['Balance', 'ROI'])
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Resumen de resultados")
    st.dataframe(styled_resumen, hide_index=True, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Columnas finales a mostrar
    final_cols = [
        "Bet?", "Bet for", "Amount", "Won?", "PNL", "Torneo", "Fecha", "Superficie",
        "Jugador 1", "J1 Puntos ATP", "Jugador 2", "J2 Puntos ATP",
        "J1 H2H %", "J2 H2H %", "J1 Rend. Reciente", "J1 Rend. Superficie", "Rend. Ultra reciente J1",
        "J2 Rend. Reciente", "J2 Rend. Superficie", "Rend. Ultra reciente J2",
        "f1", "f2", "Cuota J1", "Cuota J2", "Ganador", "ID Partido"
    ]
    
    df_display = df[final_cols].copy()
    df_display["Fecha"] = df_display["Fecha"].dt.strftime("%Y-%m-%d")
    
    vis_key = f"{prefix}_visible_cols"
    if vis_key not in st.session_state:
        st.session_state[vis_key] = saved_cfg.get("td_visible_cols", final_cols)

    with st.expander("👁️ Configurar Visibilidad de Columnas"):
        st.multiselect("Columnas Visibles", options=final_cols, key=vis_key)

    visible_cols = st.session_state[vis_key]
    visible_cols = [c for c in visible_cols if c in df_display.columns]
    
    df_to_edit = df_display[visible_cols].copy()
    
    styled_df_display = df_to_edit.style
    if "Won?" in visible_cols:
        styled_df_display = styled_df_display.map(
            lambda v: 'background-color: #d4edda; color: #155724; font-weight: bold;' if v == 'Yes' 
            else ('background-color: #f8d7da; color: #721c24; font-weight: bold;' if v == 'No' else ''), 
            subset=['Won?']
        )
    
    col_btn_1, col_btn_2 = st.columns([2, 8])
    with col_btn_1:
        if st.button("💾 Guardar Cambios en CSV", key=f"btn_save_csv_{prefix}", type="primary", use_container_width=True):
            from scripts.data_persistence import apply_edits
            success = apply_edits(f"dt_{prefix}", df_display, table_name, manager)
            if success:
                st.cache_data.clear()
                st.rerun()

    st.data_editor(
        styled_df_display, 
        key=f"dt_{prefix}",
        height=1000, 
        hide_index=True, 
        use_container_width=True,
        disabled=[c for c in final_cols if c not in ["Cuota J1", "Cuota J2", "Ganador", "Fecha"]],
        column_config={
            "ID Partido": st.column_config.NumberColumn("ID Partido", format="%d"),
            "Cuota J1": st.column_config.NumberColumn("Cuota J1", format="%.3f"),
            "Cuota J2": st.column_config.NumberColumn("Cuota J2", format="%.3f")
        }
    )

# --- BETA TAB RENDERER (Download logic included) ---

def render_bet_tab(mode, table_name, prefix, header):
    st.header(header)
    
    # 1. Refresh Button
    col_ref, _ = st.columns([2, 8])
    with col_ref:
        if st.button(f"🔄 Actualizar {mode}", key=f"btn_{prefix}_refresh", type="primary", use_container_width=True):
            st.session_state.pending_refresh_mode = mode
            st.rerun()

    # 2. Date Filters
    st.write("##### Filtrar por Fecha")
    c1, c2, c3, c4, c5 = st.columns(5)
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    next2 = today + timedelta(days=2)
    
    filter_key = f"{prefix}_date_filter"
    if filter_key not in st.session_state: st.session_state[filter_key] = str(today)
    
    with c1: 
        if st.button("Ayer", key=f"b_{prefix}_y", use_container_width=True): st.session_state[filter_key] = str(yesterday); st.rerun()
    with c2: 
        if st.button("Hoy", key=f"b_{prefix}_t", use_container_width=True): st.session_state[filter_key] = str(today); st.rerun()
    with c3: 
        if st.button("Mañana", key=f"b_{prefix}_to", use_container_width=True): st.session_state[filter_key] = str(tomorrow); st.rerun()
    with c4: 
        if st.button("+2 Días", key=f"b_{prefix}_p2", use_container_width=True): st.session_state[filter_key] = str(next2); st.rerun()
    with c5: 
        if st.button("Todos", key=f"b_{prefix}_all", use_container_width=True): st.session_state[filter_key] = "All"; st.rerun()

    # 3. Load and Filter
    df_raw = load_matches_data(table_name)
    
    if df_raw.empty:
        st.warning("No hay datos disponibles.")
        return

    # Solo partidos sin terminar
    df_f = df_raw[df_raw["Ganador"].isna() | (df_raw["Ganador"] == "-") | (df_raw["Ganador"] == "")].copy()
    
    if st.session_state[filter_key] != "All":
        target = pd.to_datetime(st.session_state[filter_key]).date()
        df_f = df_f[df_f["Fecha"].dt.date == target]
    else:
        df_f = df_f[(df_f["Fecha"].dt.date >= yesterday) & (df_f["Fecha"].dt.date <= next2)]

    if df_f.empty:
        st.info("No hay partidos programados para esta selección.")
        return

    # 4. Sorting logic (Tournament + Time)
    # Note: 'Hora' column must exist in refresh_data.py
    if "Hora" in df_f.columns:
        df_f = df_f.sort_values(by=["Torneo", "Hora"], ascending=[True, True])
    else:
        df_f = df_f.sort_values(by="Torneo")

    # 5. TSV Export Generation
    export_cols = [
        "Torneo", "Fecha", "Superficie", "Jugador 1", "J1 Key", "J1 Puntos ATP",
        "Jugador 2", "J2 Key", "J2 Puntos ATP", 
        "J1 H2H", "J1 H2H %", "J2 H2H", "J2 H2H %",
        "J1 Rend. Reciente", "J1 Rend. Superficie", "Rend. Ultra reciente J1", 
        "J2 Rend. Reciente", "J2 Rend. Superficie", "Rend. Ultra reciente J2", 
        "Cuota J1", "Cuota J2"
    ]
    
    # Pre-format for TSV (Excel compatible with commas)
    df_tsv = df_f.copy()
    if len(df_tsv) > 0:
        for c in df_tsv.columns:
            if any(x in c for x in ["Cuota", "Puntos", "Rend", "H2H", "f1", "f2"]):
                df_tsv[c] = df_tsv[c].apply(lambda x: str(x).replace(".", ",") if pd.notna(x) and x != "" else x)

        tsv_lines = ["\t".join(export_cols)]
        for _, row in df_tsv.iterrows():
            row_data = [str(row.get(c, "")) for c in export_cols]
            tsv_lines.append("\t".join(row_data))
            tsv_lines.append("") # Línea en blanco entre partidos
        
        tsv_data = "\n".join(tsv_lines)
        
        col_st, col_dl = st.columns([8, 2])
        with col_st:
            st.success(f"Mostrando {len(df_f)} partidos encontrados.")
        with col_dl:
            st.download_button(
                label="📥 Descargar TSV",
                data=tsv_data,
                file_name=f"otorongo_{prefix}_{st.session_state[filter_key]}.tsv",
                mime="text/tab-separated-values",
                use_container_width=True
            )

    # Display Dataframe
    display_df = df_f.copy()
    # Format date for display
    display_df["Fecha"] = display_df["Fecha"].dt.strftime("%d/%m/%y")
    st.dataframe(
        display_df, 
        height=1000, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "ID Partido": st.column_config.NumberColumn("ID Partido", format="%d"),
            "Cuota J1": st.column_config.NumberColumn("Cuota J1", format="%.3f"),
            "Cuota J2": st.column_config.NumberColumn("Cuota J2", format="%.3f")
        }
    )

# --- MAIN APP ---
# --- Auto-Resume Refresh Logic ---
if st.session_state.pending_refresh_mode is not None:
    mode = st.session_state.pending_refresh_mode
    st.info(f"🚀 Reanudando actualización automática de partidos {mode}...")
    prog_container = st.empty()
    try:
        from scripts.refresh_data import refresh
        with prog_container.container():
            prog_bar = st.progress(0)
            prog_text = st.empty()
        
        def update_progress(step, total, msg):
            pct = max(0, min(100, int((step / total) * 100)))
            prog_bar.progress(pct)
            prog_text.text(msg)
        
        with st.spinner(f"Actualizando datos {mode}..."):
            res = refresh(mode=mode, progress_callback=update_progress)
        
        prog_container.empty()
        
        if res and res.get("status") == "NEED_SURFACE":
            show_surface_assignment_dialog(res.get("tournaments", []), mode)
        elif res and res.get("status") == "SUCCESS":
            st.cache_data.clear()
            st.session_state.pending_refresh_mode = None
            st.rerun()
        elif res and res.get("status") == "ERROR":
            st.error(f"❌ Error en el refresh: {res.get('msg', 'Error desconocido')}")
            st.session_state.pending_refresh_mode = None
        else:
            st.error("Error desconocido en el refresh.")
            st.session_state.pending_refresh_mode = None
    except Exception as e:
        st.error(f"Error crítico en el refresh automático: {e}")
        st.session_state.pending_refresh_mode = None



def render_analytics_dashboard(prefix, df_computed, show_categories=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("📈 Evolución de rendimiento")
    st.caption(f"Los datos reflejan los parámetros configurados en la pestaña **{prefix.upper()} data** (pesos, umbrales, cuota mínima).")


    if df_computed is None or df_computed.empty:
        st.info("⚙️ Accede primero a la pestaña **" + prefix.upper() + " data** para cargar y calcular los datos. El gráfico se generará automáticamente con los parámetros que hayas configurado.")
    else:
        # --- Solo apuestas terminadas ---
        df_fin = df_computed[df_computed["Won?"].isin(["Yes", "No"])].copy()
        df_fin["Fecha"] = pd.to_datetime(df_fin["Fecha"], errors="coerce")
        df_fin = df_fin.dropna(subset=["Fecha"])

        if df_fin.empty:
            st.warning("No hay apuestas con resultado definido en el periodo seleccionado.")
        else:
            # --- Controles de la pestaña ---
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 3, 3])
            with col_ctrl1:
                freq_map = {"Día": "D", "Semana": "W", "Mes": "ME"}
                freq_label = st.radio("Agrupación", list(freq_map.keys()), index=1, horizontal=True, key=f"{prefix}_analytics_freq")
                freq = freq_map[freq_label]
            with col_ctrl2:
                metric_opts = ["Balance €", "ROI %", "% Acierto"]
                metrics_sel = st.multiselect(
                    "Métricas (máx. 2)", metric_opts, default=["Balance €"],
                    max_selections=2, key=f"{prefix}_analytics_metrics"
                )
            with col_ctrl3:
                show_acum = st.checkbox("Balance acumulado", value=False, key=f"{prefix}_analytics_acum")

            if not metrics_sel:
                st.info("Selecciona al menos una métrica para mostrar el gráfico.")
            else:
                # --- Agrupación temporal con resample ---
                df_chart = df_fin.set_index("Fecha").sort_index()

                def build_period_stats(group):
                    wins   = (group["Won?"] == "Yes").sum()
                    losses = (group["Won?"] == "No").sum()
                    total  = wins + losses
                    pnl    = group["PNL"].sum()
                    invest = group["Amount"].sum()
                    roi    = (pnl / invest * 100) if invest > 0 else 0
                    wr     = (wins / total * 100) if total > 0 else 0
                    start  = group.index.min().strftime("%d/%m/%y") if len(group) > 0 else ""
                    end    = group.index.max().strftime("%d/%m/%y") if len(group) > 0 else ""
                    return pd.Series({
                        "Balance €": pnl,
                        "ROI %": roi,
                        "% Acierto": wr,
                        "n_bets": int(total),
                        "wins": int(wins),
                        "losses": int(losses),
                        "date_range": f"{start} – {end}" if start != end else start
                    })

                df_grouped = df_chart.resample(freq).apply(build_period_stats).dropna()
                df_grouped = df_grouped[df_grouped["n_bets"] > 0]

                if df_grouped.empty:
                    st.warning("No hay suficientes datos para la agrupación seleccionada.")
                else:
                    if show_acum and "Balance €" in metrics_sel:
                        df_grouped["Balance €"] = df_grouped["Balance €"].cumsum()

                    # --- Construcción del gráfico Plotly ---
                    dual = len(metrics_sel) == 2
                    fig = make_subplots(specs=[[{"secondary_y": dual}]])

                    color_map = {"Balance €": "#00c49a", "ROI %": "#f7a600", "% Acierto": "#7b61ff"}
                    dash_map  = {"Balance €": "solid",   "ROI %": "dot",     "% Acierto": "dash"}

                    for i, metric in enumerate(metrics_sel):
                        secondary = (i == 1 and dual)
                        y_vals = df_grouped[metric]
                        x_vals = df_grouped.index

                        hover_text = [
                            f"<b>{r['date_range']}</b><br>"
                            f"Apuestas: {int(r['n_bets'])} ({int(r['wins'])}W-{int(r['losses'])}L)<br>"
                            f"Balance: {r['Balance €']:.2f} €<br>"
                            f"ROI: {r['ROI %']:.2f}%<br>"
                            f"Acierto: {r['% Acierto']:.1f}%"
                            for _, r in df_grouped.iterrows()
                        ]

                        hover_kwargs = {}
                        if i == 0:
                            hover_kwargs = dict(
                                hovertemplate="%{customdata}<extra></extra>",
                                customdata=hover_text,
                            )
                        else:
                            hover_kwargs = dict(hoverinfo="skip")

                        fig.add_trace(
                            go.Scatter(
                                x=x_vals, y=y_vals,
                                name=metric,
                                mode="lines+markers",
                                line=dict(color=color_map[metric], width=2.5, dash=dash_map[metric]),
                                marker=dict(size=6),
                                **hover_kwargs
                            ),
                            secondary_y=secondary
                        )

                    # Línea de referencia en 0 para Balance/ROI
                    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1)

                    y1_label = metrics_sel[0]
                    y2_label = metrics_sel[1] if dual else ""
                    fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=420,
                        xaxis=dict(showgrid=False, tickformat="%d %b '%y"),
                        yaxis=dict(title=y1_label, gridcolor="rgba(255,255,255,0.07)"),
                    )
                    if dual:
                        fig.update_yaxes(title_text=y2_label, secondary_y=True, gridcolor="rgba(0,0,0,0)")

                    st.plotly_chart(fig, use_container_width=True)

                    # --- Mini resumen del periodo visible ---
                    total_bets = int(df_grouped["n_bets"].sum())
                    total_bal  = df_grouped["Balance €"].iloc[-1] if show_acum and "Balance €" in metrics_sel else df_fin["PNL"].sum()
                    avg_roi    = df_grouped["ROI %"].mean()
                    avg_wr     = df_grouped["% Acierto"].mean()

                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Apuestas totales", f"{total_bets}")
                    mc2.metric("Balance final", f"{total_bal:.2f} €")
                    mc3.metric("ROI promedio", f"{avg_roi:.2f}%")
                    mc4.metric("Acierto promedio", f"{avg_wr:.1f}%")

            # --- 🚥 ANÁLISIS POR SUPERFICIE (TACTICAL TRAFFIC LIGHT) ---
            st.divider()
            st.subheader("🚥 Análisis por Superficie")
            st.caption("Identifica tu ventaja competitiva (Edge) según el tipo de pista.")
            
            df_surf_bets = df_fin[df_fin["Bet?"] == "Yes"].copy()
            if df_surf_bets.empty:
                st.info("No hay apuestas realizadas en este periodo para analizar superficies.")
            else:
                surf_metrics = []
                surfaces = df_surf_bets["Superficie"].dropna().unique()
                for s in surfaces:
                    df_s = df_surf_bets[df_surf_bets["Superficie"] == s]
                    n_bets = len(df_s)
                    if n_bets > 0:
                        wins = (df_s["Won?"] == "Yes").sum()
                        wr = (wins / n_bets * 100)
                        pnl = df_s["PNL"].sum()
                        inv = df_s["Amount"].sum()
                        roi = (pnl / inv * 100) if inv > 0 else 0
                        
                        def get_surf_odds(r):
                            return r["Cuota J1"] if norm(r["Bet for"]) == norm(r["Jugador 1"]) else r["Cuota J2"]
                            
                        odds_list = [get_surf_odds(r) for _, r in df_s.iterrows()]
                        avg_odds = sum(odds_list) / n_bets
                        
                        surf_metrics.append({
                            "Superficie": s,
                            "Apuestas": n_bets,
                            "Acierto %": wr,
                            "Balance €": pnl,
                            "ROI %": roi,
                            "Cuota Media": avg_odds
                        })
                        
                if surf_metrics:
                    df_sm = pd.DataFrame(surf_metrics)
                    
                    col_radar, col_table = st.columns([1, 2])
                    with col_radar:
                        colors = ["#00c49a" if r > 0 else "#dc3545" for r in df_sm["ROI %"]]
                        fig_bar = go.Figure(go.Bar(
                            x=df_sm["ROI %"],
                            y=df_sm["Superficie"],
                            orientation='h',
                            marker_color=colors,
                            text=[f"{r:.1f}%" for r in df_sm["ROI %"]],
                            textposition="auto"
                        ))
                        fig_bar.update_layout(
                            template="plotly_dark",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=0, r=30, t=20, b=20),
                            height=250,
                            xaxis=dict(title="ROI %", showgrid=False, zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                            yaxis=dict(showgrid=False)
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                        
                    with col_table:
                        df_table = df_sm.copy()
                        df_table["Apuestas"] = df_table["Apuestas"].apply(lambda x: f"{x} (⚠️ Muestra pequeña)" if x < 10 else str(x))
                        df_table["Acierto %"] = df_table["Acierto %"].apply(lambda x: f"{x:.1f}%")
                        df_table["Balance €"] = df_table["Balance €"].apply(lambda x: f"{x:.2f} €")
                        df_table["ROI %"] = df_table["ROI %"].apply(lambda x: f"{x:.2f}%")
                        df_table["Cuota Media"] = df_table["Cuota Media"].apply(lambda x: f"{x:.2f}")
                        
                        st.dataframe(df_table, use_container_width=True, hide_index=True)

            # --- 📊 DISTRIBUCIÓN POR RANGOS DE CUOTAS (ODDS BUCKETS & EDGE) ---
            st.divider()
            st.subheader("📊 Análisis de Rangos de Cuotas (Edge)")
            st.caption("Identifica en qué rangos de cuotas el modelo encuentra mayor valor real (Edge).")
            
            if df_surf_bets.empty:
                st.info("No hay apuestas realizadas en este periodo para analizar rangos de cuotas.")
            else:
                def get_taken_odds(row):
                    return row["Cuota J1"] if norm(row["Bet for"]) == norm(row["Jugador 1"]) else row["Cuota J2"]
                
                df_odds = df_surf_bets.copy()
                df_odds["Cuota_Tomada"] = df_odds.apply(get_taken_odds, axis=1)
                
                def assign_bucket(c):
                    if c <= 1.30: return "[1.00 - 1.30] (Fav. Extremos)"
                    elif c <= 1.65: return "[1.31 - 1.65] (Fav. Sólidos)"
                    elif c <= 2.00: return "[1.66 - 2.00] (Igualados)"
                    elif c <= 3.00: return "[2.01 - 3.00] (Underdogs)"
                    else: return "[3.01+] (Sorpresas)"
                    
                df_odds["Bucket"] = df_odds["Cuota_Tomada"].apply(assign_bucket)
                bucket_order = [
                    "[1.00 - 1.30] (Fav. Extremos)",
                    "[1.31 - 1.65] (Fav. Sólidos)",
                    "[1.66 - 2.00] (Igualados)",
                    "[2.01 - 3.00] (Underdogs)",
                    "[3.01+] (Sorpresas)"
                ]
                
                bucket_stats = []
                for b in bucket_order:
                    df_b = df_odds[df_odds["Bucket"] == b]
                    n_bets = len(df_b)
                    if n_bets > 0:
                        wins = (df_b["Won?"] == "Yes").sum()
                        wr_real = (wins / n_bets) * 100
                        wr_esp = (1 / df_b["Cuota_Tomada"]).mean() * 100
                        edge = wr_real - wr_esp
                        pnl = df_b["PNL"].sum()
                        inv = df_b["Amount"].sum()
                        yield_pct = (pnl / inv) * 100 if inv > 0 else 0
                        
                        warning = "⚠️ < 20 apuestas" if n_bets < 20 else ""
                        display_name = f"{b}<br><i>{warning}</i>" if warning else b
                        
                        bucket_stats.append({
                            "Bucket": b,
                            "Display": display_name,
                            "Apuestas": n_bets,
                            "WR_Real": wr_real,
                            "WR_Esp": wr_esp,
                            "Edge": edge,
                            "Yield": yield_pct,
                            "Warning": warning
                        })
                        
                if bucket_stats:
                    df_bs = pd.DataFrame(bucket_stats)
                    
                    fig_buckets = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    bar_colors = ["#00c49a" if e > 0 else "#dc3545" for e in df_bs["Edge"]]
                    
                    hover_text = [
                        f"<b>{r['Bucket']}</b><br>"
                        f"Apuestas: {r['Apuestas']} {r['Warning']}<br>"
                        f"WR Real: {r['WR_Real']:.1f}%<br>"
                        f"WR Esperado: {r['WR_Esp']:.1f}%<br>"
                        f"Edge: {r['Edge']:.2f}%<br>"
                        f"Yield: {r['Yield']:.2f}%"
                        for _, r in df_bs.iterrows()
                    ]
                    
                    # Hover info for specific trace
                    hover_kwargs_bar = dict(hovertemplate="%{customdata}<extra></extra>", customdata=hover_text)
                    hover_kwargs_line = dict(hoverinfo="skip") # Skip to avoid double tooltips in unified mode
                    
                    fig_buckets.add_trace(go.Bar(
                        x=df_bs["Display"],
                        y=df_bs["Apuestas"],
                        name="Volumen de Apuestas",
                        marker_color=bar_colors,
                        opacity=0.8,
                        **hover_kwargs_bar
                    ), secondary_y=False)
                    
                    fig_buckets.add_trace(go.Scatter(
                        x=df_bs["Display"],
                        y=df_bs["Yield"],
                        name="Yield (ROI) %",
                        mode="lines+markers+text",
                        line=dict(color="#f7a600", width=3),
                        marker=dict(size=8, color="#f7a600"),
                        text=[f"{y:.1f}%" for y in df_bs["Yield"]],
                        textposition="top center",
                        textfont=dict(color="#f7a600", size=11),
                        **hover_kwargs_line
                    ), secondary_y=True)
                    
                    fig_buckets.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=350,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(title="N° Apuestas", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        hovermode="x unified"
                    )
                    
                    fig_buckets.update_yaxes(title_text="Yield %", secondary_y=True, showgrid=False)
                    
                    st.plotly_chart(fig_buckets, use_container_width=True)

            if show_categories:
        # --- 🏆 RENDIMIENTO POR CATEGORÍA DE TORNEO ---
                st.divider()
                st.subheader("🏆 Rendimiento por Categoría de Torneo")
                st.caption("Comparativa del rendimiento del modelo según la importancia del evento.")
                
                if df_surf_bets.empty:
                    st.info("No hay apuestas realizadas en este periodo para analizar categorías de torneo.")
                else:
                    cat_file_path = "data/tournament_categories.json"
                    default_categories = {
                        "Grand Slam": ["Australian Open", "Roland Garros", "French Open", "Wimbledon", "US Open"],
                        "Masters 1000": ["Indian Wells", "Miami", "Monte Carlo", "Madrid", "Rome", "Montreal", "Toronto", "Cincinnati", "Shanghai", "Paris"],
                        "Copa Davis": ["Davis Cup", "Copa Davis"]
                    }
                    
                    if not os.path.exists("data"):
                        os.makedirs("data")
                    if not os.path.exists(cat_file_path):
                        with open(cat_file_path, "w") as f:
                            json.dump(default_categories, f, indent=4)
                        tourney_cats = default_categories
                    else:
                        try:
                            with open(cat_file_path, "r") as f:
                                tourney_cats = json.load(f)
                        except Exception:
                            tourney_cats = default_categories
                            
                    def classify_tournament(name):
                        name_lower = str(name).lower()
                        for cat_name, keywords in tourney_cats.items():
                            for kw in keywords:
                                if kw.lower() in name_lower:
                                    return cat_name
                        return "ATP 500 / 250"
                        
                    df_cat_bets = df_surf_bets.copy()
                    df_cat_bets["Categoría"] = df_cat_bets["Torneo"].apply(classify_tournament)
                    
                    cat_stats = []
                    for cat in df_cat_bets["Categoría"].unique():
                        df_c = df_cat_bets[df_cat_bets["Categoría"] == cat]
                        n_bets = len(df_c)
                        if n_bets > 0:
                            wins = (df_c["Won?"] == "Yes").sum()
                            wr_real = (wins / n_bets) * 100
                            
                            def get_taken_odds(r):
                                return r["Cuota J1"] if norm(r["Bet for"]) == norm(r["Jugador 1"]) else r["Cuota J2"]
                                
                            cuotas = df_c.apply(get_taken_odds, axis=1)
                            wr_esp = (1 / cuotas).mean() * 100
                            edge = wr_real - wr_esp
                            
                            pnl = df_c["PNL"].sum()
                            inv = df_c["Amount"].sum()
                            yield_pct = (pnl / inv) * 100 if inv > 0 else 0
                            
                            cat_stats.append({
                                "Categoría": cat,
                                "Apuestas": n_bets,
                                "WR_Real": wr_real,
                                "Edge": edge,
                                "Yield": yield_pct,
                                "Balance": pnl
                            })
                            
                    if cat_stats:
                        df_cs = pd.DataFrame(cat_stats)
                        df_cs = df_cs.sort_values("Apuestas", ascending=False)
                        
                        cat_metric = st.radio(
                            "Visualizar métrica en el gráfico:", ["Volumen de Apuestas", "Balance General (€)"], 
                            horizontal=True, key=f"{prefix}_analytics_cat_metric"
                        )
                        
                        col_chart, col_cat_table = st.columns(2)
                        with col_chart:
                            if cat_metric == "Volumen de Apuestas":
                                fig = go.Figure(data=[go.Pie(
                                    labels=df_cs["Categoría"],
                                    values=df_cs["Apuestas"],
                                    hole=0.4,
                                    textinfo='label+percent',
                                    marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                                )])
                                fig.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=20, r=20, t=20, b=20),
                                    height=250,
                                    showlegend=False
                                )
                            else:
                                # Gráfico de Barras para Balance
                                fig = go.Figure(go.Bar(
                                    x=df_cs["Categoría"],
                                    y=df_cs["Balance"],
                                    marker_color=["#00c49a" if b > 0 else "#dc3545" for b in df_cs["Balance"]],
                                    text=[f"{b:.2f}€" for b in df_cs["Balance"]],
                                    textposition="auto"
                                ))
                                fig.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=20, r=20, t=20, b=20),
                                    height=250,
                                    xaxis=dict(title="Categoría"),
                                    yaxis=dict(title="Balance €", zeroline=True, zerolinecolor="rgba(255,255,255,0.2)")
                                )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col_cat_table:
                            df_cat_table = df_cs.copy()
                            df_cat_table["WR Real"] = df_cat_table["WR_Real"].apply(lambda x: f"{x:.1f}%")
                            df_cat_table["Edge"] = df_cat_table["Edge"].apply(lambda x: f"{x:.2f}%")
                            df_cat_table["Yield"] = df_cat_table["Yield"].apply(lambda x: f"{x:.2f}%")
                            df_cat_table["Balance €"] = df_cat_table["Balance"].apply(lambda x: f"{x:.2f} €")
                            df_cat_table = df_cat_table.drop(columns=["WR_Real", "Balance"])
                            
                            st.dataframe(df_cat_table, use_container_width=True, hide_index=True)

                # --- ☠️ RADIOGRAFÍA DE RIESGO (DRAWDOWN & SUPERVIVENCIA) ---
            st.divider()
            st.subheader("☠️ Radiografía de Riesgo")
            st.caption("Métricas de supervivencia y gestión de bankroll para la estrategia ATP.")

            if df_surf_bets.empty:
                st.info("No hay apuestas realizadas en este periodo para calcular el riesgo.")
            else:
                bankroll = st.number_input(
                    "Bankroll inicial (€)", min_value=100, value=500, step=50,
                    key=f"{prefix}_analytics_bankroll",
                    help="Introduce tu bankroll de referencia para calcular el Risk of Ruin y el Stress Test."
                )

                # --- Cálculo del Drawdown ---
                df_risk = df_surf_bets.sort_values("Fecha").reset_index(drop=True)
                df_risk["PNL_cum"]  = df_risk["PNL"].cumsum()
                df_risk["Peak"]     = df_risk["PNL_cum"].cummax()
                df_risk["Drawdown"] = df_risk["PNL_cum"] - df_risk["Peak"]
                max_dd      = df_risk["Drawdown"].min()
                current_dd  = df_risk["Drawdown"].iloc[-1]

                # --- KPIs ---
                # 1. Profit Factor
                gains  = df_surf_bets[df_surf_bets["PNL"] > 0]["PNL"].sum()
                losses = df_surf_bets[df_surf_bets["PNL"] < 0]["PNL"].sum()
                if losses == 0:
                    profit_factor_str = "∞"
                else:
                    profit_factor_str = f"{gains / abs(losses):.2f}"

                # 2. Racha Máxima de Pérdidas
                max_streak = cur_streak = 0
                for result in df_risk["Won?"].values:
                    if result == "No":
                        cur_streak += 1
                        max_streak = max(max_streak, cur_streak)
                    else:
                        cur_streak = 0

                # 3. Time to Recovery (apuestas promedio para salir de un drawdown)
                recovery_counts = []
                in_drawdown = False
                dd_start_idx = 0
                for i, row in df_risk.iterrows():
                    if not in_drawdown and row["Drawdown"] < 0:
                        in_drawdown = True
                        dd_start_idx = i
                    elif in_drawdown and row["Drawdown"] == 0:
                        recovery_counts.append(i - dd_start_idx)
                        in_drawdown = False
                avg_recovery = int(round(sum(recovery_counts) / len(recovery_counts))) if recovery_counts else "N/A"

                # 4. Risk of Ruin (Gambler's Ruin approximation)
                total_bets_risk = len(df_surf_bets)
                wins_risk       = (df_surf_bets["Won?"] == "Yes").sum()
                wr_risk         = wins_risk / total_bets_risk if total_bets_risk > 0 else 0
                avg_stake       = df_surf_bets["Amount"].mean() if total_bets_risk > 0 else 1
                net_edge        = wr_risk - (1 - wr_risk)

                if net_edge <= 0:
                    ror_pct = 100.0
                else:
                    units = bankroll / avg_stake
                    ror_base = (1 - net_edge) / (1 + net_edge)
                    ror_pct = min((ror_base ** units) * 100, 100.0)

                ror_color = "#dc3545" if ror_pct > 20 else ("#f7a600" if ror_pct > 5 else "#00c49a")

                # --- Layout: 4 KPIs ---
                rk1, rk2, rk3, rk4 = st.columns(4)
                rk1.metric("Profit Factor", profit_factor_str, help="Ganancias totales / Pérdidas totales absolutas. >1 = rentable.")
                rk2.metric("Racha Máxima", f"{max_streak} derrotas", help="Máximo número de apuestas perdidas consecutivamente.")
                rk3.metric("Time to Recovery", f"{avg_recovery} apuestas" if recovery_counts else "N/A", help="Apuestas promedio para recuperar un drawdown.")
                rk4.metric("Risk of Ruin", f"{ror_pct:.2f}%", help="Probabilidad estadística de perder el bankroll completo.")

                # --- Gráfico de Drawdown (Estalactitas) ---
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=df_risk["Fecha"],
                    y=df_risk["Drawdown"],
                    fill="tozeroy",
                    fillcolor="rgba(220, 53, 69, 0.20)",
                    line=dict(color="#dc3545", width=1.5),
                    name="Drawdown (€)",
                    hovertemplate="<b>%{x|%d %b '%y}</b><br>Drawdown: %{y:.2f} €<extra></extra>"
                ))
                fig_dd.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1)
                fig_dd.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=20, b=0),
                    height=260,
                    yaxis=dict(
                        title="Drawdown (€)",
                        range=[max_dd * 1.15, 0],
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.05)"
                    ),
                    xaxis=dict(showgrid=False),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_dd, use_container_width=True)

                # --- Termómetro de Riesgo ---
                thermo_pct = (abs(current_dd) / abs(max_dd) * 100) if max_dd != 0 else 0
                if thermo_pct < 30:
                    thermo_color = "#00c49a"
                elif thermo_pct < 70:
                    thermo_color = "#f7a600"
                else:
                    thermo_color = "#dc3545"

                st.markdown(
                    f"""
                    <div style='margin-bottom:6px;'>
                        <span style='font-size:0.85rem; color:#aaa;'>🌡️ Drawdown Actual vs Máximo Histórico</span>
                        &nbsp;&nbsp;
                        <span style='color:#ccc; font-size:0.85rem;'>
                            <b style='color:{thermo_color};'>{current_dd:.2f} €</b> &nbsp;/&nbsp; Máximo: <b>{max_dd:.2f} €</b>
                        </span>
                    </div>
                    <div style='background:#1e1e2e; border-radius:8px; height:14px; width:100%; overflow:hidden;'>
                        <div style='background:{thermo_color}; width:{thermo_pct:.1f}%; height:100%; border-radius:8px; transition: width 0.4s ease;'></div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # --- Stress Test ---
                if abs(max_dd) > bankroll * 0.5:
                    st.warning(
                        f"⚠️ **Stress Test:** Tu máximo drawdown histórico ({abs(max_dd):.2f} €) supera el 50% "
                        f"de tu bankroll declarado ({bankroll:.0f} €). Tu stake actual es demasiado agresivo "
                        f"para tu volatilidad histórica. Considera reducir el riesgo."
                    )

            # --- 🎯 AUDITORÍA DE CALIBRACIÓN (CURVA DE FIABILIDAD) ---
            st.divider()
            st.subheader("🎯 Auditoría de Calibración")
            st.caption("¿Las probabilidades del modelo coinciden con los resultados reales? Cuanto más cerca estés de la diagonal, más fiable es el modelo.")

            # Usar TODA la muestra: todos los partidos con ganador definido (no solo apuestas)
            df_cal_raw = df_computed.copy()
            df_cal_raw["Ganador_norm"] = df_cal_raw["Ganador"].apply(norm)
            # Filtrar solo partidos con resultado real definido (sin retiros/cancelados/vacíos)
            _invalid = ["retired", "cancelled", "walkover", "retirado", "cancelado", "-", ""]
            df_cal = df_cal_raw[
                df_cal_raw["Ganador_norm"].notna() &
                ~df_cal_raw["Ganador_norm"].isin(["", "-"]) &
                ~df_cal_raw["Ganador_norm"].str.contains("|".join(["retired","cancelled","walkover","retirado","cancelado"]), na=True)
            ].copy()

            # La probabilidad a auditar: siempre la del jugador más favorecido (max(f1, f2))
            df_cal["f_model"] = df_cal[["f1", "f2"]].max(axis=1) / 100.0

            # El resultado real: 1 si el jugador favorito ganó, 0 si perdió
            df_cal["fav_is_j1"] = df_cal["f1"] >= df_cal["f2"]
            df_cal["fav_name"]  = df_cal.apply(lambda r: r["Jugador 1"] if r["fav_is_j1"] else r["Jugador 2"], axis=1)
            df_cal["outcome"]   = (df_cal.apply(lambda r: norm(r["Ganador"]) == norm(r["fav_name"]), axis=1)).astype(int)

            if len(df_cal) < 10:
                st.info("Se necesitan al menos 10 partidos con resultado para calcular la curva de calibración.")
            else:
                # --- Brier Score ---
                brier = ((df_cal["f_model"] - df_cal["outcome"]) ** 2).mean()
                brier_delta = "✅ Excelente" if brier <= 0.20 else ("⚠️ Aceptable" if brier <= 0.25 else "❌ Bajo")

                cal_c1, cal_c2 = st.columns(2)
                cal_c1.metric("Brier Score", f"{brier:.4f}", help="Cuanto más cercano a 0, mejor calibrado. Referencia: ≤0.20 = excelente, 0.25 = modelo aleatorio.")
                cal_c2.metric("Valoración", brier_delta)

                # --- Calibration Buckets (5% bins de 50% a 100%) ---
                bins   = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
                labels = ["50-55%", "55-60%", "60-65%", "65-70%", "70-75%",
                          "75-80%", "80-85%", "85-90%", "90-95%", "95-100%"]

                df_cal_hi = df_cal[df_cal["f_model"] >= 0.50].copy()
                df_cal_hi["bucket"] = pd.cut(df_cal_hi["f_model"], bins=bins, labels=labels, right=False)

                bucket_data = []
                for lbl in labels:
                    df_b = df_cal_hi[df_cal_hi["bucket"] == lbl]
                    if len(df_b) > 0:
                        bucket_data.append({
                            "Bucket": lbl,
                            "Prob Media": df_b["f_model"].mean() * 100,
                            "WR Real": df_b["outcome"].mean() * 100,
                            "N": len(df_b)
                        })

                if bucket_data:
                    df_bk = pd.DataFrame(bucket_data)
                    max_n  = df_bk["N"].max()

                    fig_cal = go.Figure()

                    # Línea de calibración perfecta (diagonal)
                    fig_cal.add_trace(go.Scatter(
                        x=[50, 100], y=[50, 100],
                        mode="lines",
                        line=dict(dash="dot", color="rgba(255,255,255,0.35)", width=1.5),
                        name="Calibración Perfecta",
                        hoverinfo="skip"
                    ))

                    # Burbujas de calibración
                    bubble_colors = [
                        "#00c49a" if abs(r["WR Real"] - r["Prob Media"]) <= 5 else "#f7a600" if abs(r["WR Real"] - r["Prob Media"]) <= 12 else "#dc3545"
                        for _, r in df_bk.iterrows()
                    ]
                    fig_cal.add_trace(go.Scatter(
                        x=df_bk["Prob Media"],
                        y=df_bk["WR Real"],
                        mode="markers+text",
                        marker=dict(
                            size=df_bk["N"],
                            sizemode="area",
                            sizeref=2.0 * max_n / (40 ** 2),
                            sizemin=8,
                            color=bubble_colors,
                            opacity=0.8,
                            line=dict(color="white", width=1)
                        ),
                        text=df_bk["Bucket"],
                        textposition="top center",
                        textfont=dict(size=10, color="rgba(255,255,255,0.7)"),
                        customdata=df_bk[["N", "WR Real", "Prob Media"]].values,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            "N° Partidos: %{customdata[0]}<br>"
                            "WR Real: %{customdata[1]:.1f}%<br>"
                            "Prob. Modelo: %{customdata[2]:.1f}%<br>"
                            "Diferencial: %{customdata[1]:.1f}% vs %{customdata[2]:.1f}%"
                            "<extra></extra>"
                        ),
                        name="Buckets de Calibración"
                    ))

                    fig_cal.update_layout(
                        template="plotly_dark",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=20, b=0),
                        height=380,
                        xaxis=dict(title="Probabilidad del Modelo (%)", range=[48, 102], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(title="Win Rate Real (%)", range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)

                    # --- Insight Automático ---
                    min_prob_threshold = st.session_state.get(f"{prefix}_hard_min_prob", 55.0)
                    # Buscar bucket más cercano al umbral actual
                    threshold_bucket = df_bk.iloc[(df_bk["Prob Media"] - min_prob_threshold).abs().argsort()[:1]]
                    if not threshold_bucket.empty:
                        tb = threshold_bucket.iloc[0]
                        diff = tb["WR Real"] - tb["Prob Media"]
                        if diff < -5:
                            st.warning(
                                f"⚠️ **Sesgo Optimista detectado** en el rango ~{tb['Bucket']}: "
                                f"El modelo predice **{tb['Prob Media']:.1f}%** de victoria, pero el Win Rate real es **{tb['WR Real']:.1f}%**. "
                                f"El modelo sobreestima las victorias en este rango. "
                                f"**Sugerencia:** Sube el umbral de entrada para mayor seguridad."
                            )
                        elif diff > 5:
                            st.success(
                                f"✅ **Sesgo Conservador** en el rango ~{tb['Bucket']}: "
                                f"El modelo predice **{tb['Prob Media']:.1f}%** pero el Win Rate real es **{tb['WR Real']:.1f}%**. "
                                f"El modelo está siendo conservador. Podrías bajar ligeramente el umbral para capturar más valor."
                            )
                        else:
                            st.success(
                                f"✅ **Bien calibrado** en el rango de entrada (~{tb['Bucket']}): "
                                f"El modelo predice **{tb['Prob Media']:.1f}%** y el Win Rate real es **{tb['WR Real']:.1f}%**. "
                                f"El umbral actual parece ajustado."
                            )

            # --- 👤 FICHA DE ANÁLISIS DE JUGADOR (MIRROR MODE) ---
            st.divider()
            st.subheader("👤 Ficha de Análisis de Jugador")
            st.caption("Analiza la rentabilidad de apostar a favor o en contra de un jugador específico.")

            # Filtrar df_fin para solo partidos donde hubo apuesta
            df_bets = df_fin[df_fin["Bet?"] == "Yes"].copy()
            
            if df_bets.empty:
                st.info("No hay apuestas realizadas en el periodo seleccionado para analizar jugadores.")
            else:
                # Obtener lista única de jugadores involucrados en apuestas
                players_with_bets = set(df_bets["Jugador 1"]).union(set(df_bets["Jugador 2"]))
                players_sorted = sorted(list(players_with_bets))
                
                selected_player = st.selectbox("Buscar Jugador:", [""] + players_sorted, index=0, key=f"{prefix}_analytics_player_sel")
                
                if selected_player:
                    # df donde el jugador participó y hubo apuesta
                    df_player = df_bets[(df_bets["Jugador 1"] == selected_player) | (df_bets["Jugador 2"] == selected_player)].copy()
                    
                    if not df_player.empty:
                        # Calcular ROI total del jugador
                        total_pnl = df_player["PNL"].sum()
                        total_invest = df_player["Amount"].sum()
                        total_roi = (total_pnl / total_invest * 100) if total_invest > 0 else 0
                        
                        # Badge Dinámico
                        if len(df_player) < 15:
                            badge_color = "#f7a600"
                            badge_text = "🟡 Muestra Pequeña"
                        elif total_roi > 5:
                            badge_color = "#00c49a"
                            badge_text = "🟢 Rentable"
                        elif total_roi < -5:
                            badge_color = "#dc3545"
                            badge_text = "🔴 Evitar"
                        else:
                            badge_color = "#f7a600"
                            badge_text = "🟡 Neutro"
                            
                        st.markdown(f"**ROI Total del Jugador:** <span style='color:{badge_color}; font-weight:bold;'>{total_roi:.2f}%</span> &nbsp; | &nbsp; **{badge_text}**", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # --- Bloque Mirror (2 Columnas) ---
                        col_back, col_lay = st.columns(2)
                        
                        df_back = df_player[df_player["Bet for"] == selected_player]
                        df_lay = df_player[df_player["Bet for"] != selected_player]
                        
                        def calc_mirror_metrics(df_sub):
                            n_bets = len(df_sub)
                            wins = (df_sub["Won?"] == "Yes").sum()
                            wr = (wins / n_bets * 100) if n_bets > 0 else 0
                            
                            def get_odds(row):
                                return row["Cuota J1"] if norm(row["Bet for"]) == norm(row["Jugador 1"]) else row["Cuota J2"]
                            
                            if n_bets > 0:
                                odds_list = [get_odds(r) for _, r in df_sub.iterrows()]
                                avg_odds = sum(odds_list) / n_bets
                            else:
                                avg_odds = 0
                                
                            pnl = df_sub["PNL"].sum()
                            inv = df_sub["Amount"].sum()
                            roi = (pnl / inv * 100) if inv > 0 else 0
                            return n_bets, wr, avg_odds, pnl, roi
                            
                        back_n, back_wr, back_odds, back_pnl, back_roi = calc_mirror_metrics(df_back)
                        lay_n, lay_wr, lay_odds, lay_pnl, lay_roi = calc_mirror_metrics(df_lay)
                        
                        with col_back:
                            st.markdown("#### 🔼 A Favor (Back)")
                            st.markdown(f"- **Apuestas:** {back_n}")
                            st.markdown(f"- **Win Rate:** {back_wr:.1f}%")
                            st.markdown(f"- **Cuota Media:** {back_odds:.2f}")
                            
                            back_pnl_col = '#00c49a' if back_pnl > 0 else '#dc3545'
                            back_roi_col = '#00c49a' if back_roi > 0 else '#dc3545'
                            st.markdown(f"- **Balance:** <span style='color: {back_pnl_col}'>{back_pnl:.2f} €</span>", unsafe_allow_html=True)
                            st.markdown(f"- **ROI:** <span style='color: {back_roi_col}'>{back_roi:.2f}%</span>", unsafe_allow_html=True)
                            
                        with col_lay:
                            st.markdown("#### 🔽 En Contra (Lay)")
                            st.markdown(f"- **Apuestas:** {lay_n}")
                            st.markdown(f"- **Win Rate (del modelo):** {lay_wr:.1f}%")
                            st.markdown(f"- **Cuota Media:** {lay_odds:.2f}")
                            
                            lay_pnl_col = '#00c49a' if lay_pnl > 0 else '#dc3545'
                            lay_roi_col = '#00c49a' if lay_roi > 0 else '#dc3545'
                            st.markdown(f"- **Balance:** <span style='color: {lay_pnl_col}'>{lay_pnl:.2f} €</span>", unsafe_allow_html=True)
                            st.markdown(f"- **ROI:** <span style='color: {lay_roi_col}'>{lay_roi:.2f}%</span>", unsafe_allow_html=True)
                            
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # --- Visualización (Abajo del Mirror) ---
                        col_surf, col_spark = st.columns(2)
                        
                        with col_surf:
                            st.markdown("##### Semáforo de Superficie")
                            surf_stats = []
                            for surf in ["Hard", "Clay", "Grass"]:
                                df_s = df_player[df_player["Superficie"] == surf]
                                if not df_s.empty:
                                    s_pnl = df_s["PNL"].sum()
                                    s_inv = df_s["Amount"].sum()
                                    s_roi = (s_pnl / s_inv * 100) if s_inv > 0 else 0
                                    surf_stats.append({"Surface": surf, "ROI": s_roi})
                            
                            if surf_stats:
                                df_surf = pd.DataFrame(surf_stats)
                                colors = ["#00c49a" if r > 0 else "#dc3545" for r in df_surf["ROI"]]
                                fig_surf = go.Figure(go.Bar(
                                    x=df_surf["ROI"],
                                    y=df_surf["Surface"],
                                    orientation='h',
                                    marker_color=colors,
                                    text=[f"{r:.1f}%" for r in df_surf["ROI"]],
                                    textposition="auto"
                                ))
                                fig_surf.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    margin=dict(l=0, r=0, t=10, b=0),
                                    height=200,
                                    xaxis=dict(title="ROI %", showgrid=False)
                                )
                                st.plotly_chart(fig_surf, use_container_width=True)
                            else:
                                st.info("No hay datos de superficie.")
                                
                        with col_spark:
                            st.markdown("##### Evolución de Profit")
                            df_spark = df_player.sort_values("Fecha")
                            df_spark["Cum_PNL"] = df_spark["PNL"].cumsum()
                            
                            fig_spark = go.Figure(go.Scatter(
                                x=df_spark["Fecha"],
                                y=df_spark["Cum_PNL"],
                                mode="lines",
                                line=dict(color="#f7a600", width=3),
                                fill='tozeroy',
                                fillcolor="rgba(247, 166, 0, 0.1)"
                            ))
                            fig_spark.update_layout(
                                template="plotly_dark",
                                plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0, r=0, t=10, b=0),
                                height=200,
                                xaxis=dict(showgrid=False),
                                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
                            )
                            st.plotly_chart(fig_spark, use_container_width=True)
                            
                        # --- Historial Reciente ---
                        st.markdown("##### 📜 Últimos 5 partidos (Apuestas)")
                        df_hist = df_player.sort_values("Fecha", ascending=False).head(5)
                        
                        for _, row in df_hist.iterrows():
                            if norm(row["Jugador 1"]) == norm(selected_player):
                                opp = row["Jugador 2"]
                            else:
                                opp = row["Jugador 1"]
                                
                            odds = row["Cuota J1"] if norm(row["Bet for"]) == norm(row["Jugador 1"]) else row["Cuota J2"]
                            
                            if row["Won?"] == "Yes":
                                res_icon = "✅"
                                pnl_color = "#00c49a"
                            else:
                                res_icon = "❌"
                                pnl_color = "#dc3545"
                                
                            bet_type = "A favor" if norm(row["Bet for"]) == norm(selected_player) else "En contra"
                            
                            st.markdown(
                                f"{res_icon} **{row['Fecha'].strftime('%d/%m/%y')}** vs {opp} &nbsp; | &nbsp; "
                                f"Apuesta: **{bet_type}** &nbsp; | &nbsp; Cuota: **{odds:.2f}** &nbsp; | &nbsp; "
                                f"PNL: <span style='color:{pnl_color}; font-weight:bold;'>{row['PNL']:.2f} €</span>",
                                unsafe_allow_html=True
                            )


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ATP data", "ATP Bet", "Challenger data", "Challenger Bet", "ATP Análisis", "Challenger Análisis"])

with tab1:
    render_stats_analysis_tab("🎾 ATP History & Analysis", "atp_matches", "data/analysis_config.json", "atp", "ATP", "🔄 Refresh ATP")
with tab2:
    render_bet_tab("ATP", "atp_matches", "atp_bet", "📅 ATP Scheduled Matches")
with tab3:
    render_stats_analysis_tab("🎾 Challenger History & Analysis", "challenger_matches", "data/challenger_config.json", "cha", "CHA", "🔄 Refresh Challenger")
with tab4:
    render_bet_tab("CHA", "challenger_matches", "cha_bet", "📅 Challenger Scheduled Matches")

with tab5:
    df_atp = st.session_state.get("atp_computed_df", None)
    render_analytics_dashboard("atp", df_atp, show_categories=True)

with tab6:
    df_cha = st.session_state.get("cha_computed_df", None)
    render_analytics_dashboard("cha", df_cha, show_categories=False)
