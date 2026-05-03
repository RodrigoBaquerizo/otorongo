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
        if not df.empty and "Fecha" in df.columns:
            df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
        return df
    except Exception as e:
        logging.error(f"Error loading {table_name}: {e}")
        return pd.DataFrame()

def load_analysis_config_v2(config_file):
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_analysis_config_v2(config_file, prefix):
    keys = ["max_bet", "min_bet", "min_odds", "min_atp", "weight_h2h", "weight_recent", "weight_surface", "weight_ranking", "weight_ultra", "min_prob", "visible_cols"]
    config = {}
    for k in keys:
        session_key = f"{prefix}_{k}"
        if session_key in st.session_state:
            config[f"td_{k}"] = st.session_state[session_key]
    
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
                
                manager.save_table("tournaments", df_final)
                
                st.success("✅ Superficies guardadas correctamente. Reanudando actualización...")
                st.info("Iniciando descarga de partidos...")
                
                # 4. Reanudar refresh (el DataManager no usa caché interno, así que leerá de Supabase directo)
                from scripts.refresh_data import refresh
                with st.spinner("Continuando actualización..."):
                    refresh(mode=mode)
                
                # 5. Limpiar caché de Streamlit DESPUÉS del refresh para que la App principal cargue los nuevos partidos
                st.cache_data.clear()
                
                # 6. Recargar la interfaz (solo ocurre después de que el refresh terminó)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error al guardar: {e}")

# --- TABS RENDERERS ---

def render_stats_analysis_tab(header, table_name, config_path, prefix, mode, button_label):
    st.header(header)
    
    # --- Configuración Persistence ---
    saved_cfg = load_analysis_config_v2(config_path)
    
    conf_map = {
        "max_bet": 100.0, "min_bet": 20.0, "min_odds": 1.08, "min_atp": 0,
        "weight_h2h": 1.0, "weight_recent": 15.0, "weight_surface": 43.8, 
        "weight_ranking": 40.5, "weight_ultra": 0.0, "min_prob": 64.0
    }
    
    for k, default in conf_map.items():
        session_key = f"{prefix}_{k}"
        if session_key not in st.session_state:
            st.session_state[session_key] = saved_cfg.get(f"td_{k}", default)

    # --- Acciones ---
    prog_container = st.empty()
    col_ref, _ = st.columns([2, 8])
    with col_ref:
        if st.button(button_label, key=f"btn_{prefix}_refresh", type="primary", use_container_width=True):
            try:
                from scripts.refresh_data import refresh
                
                with prog_container.container():
                    prog_bar = st.progress(0)
                    prog_text = st.empty()
                
                def update_progress(step, total, msg):
                    pct = max(0, min(100, int((step / total) * 100)))
                    prog_bar.progress(pct)
                    prog_text.text(msg)
                
                with st.spinner("Actualizando datos..."):
                    res = refresh(mode=mode, progress_callback=update_progress)
                
                prog_container.empty()
                
                if res and res.get("status") == "NEED_SURFACE":
                    show_surface_assignment_dialog(res.get("tournaments", []), mode)
                else:
                    st.cache_data.clear()
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

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

    # --- UI Parámetros (3 columnas) ---
    col_ui1, col_ui2, col_ui3 = st.columns(3)
    with col_ui1:
        st.markdown("##### Constantes")
        st.number_input("Apuesta máxima (€)", value=float(st.session_state[f"{prefix}_max_bet"]), step=10.0, key=f"{prefix}_max_bet")
        st.number_input("Apuesta mínima (€)", value=float(st.session_state[f"{prefix}_min_bet"]), step=5.0, key=f"{prefix}_min_bet")
        st.number_input("Cuota mínima", value=float(st.session_state[f"{prefix}_min_odds"]), step=0.01, key=f"{prefix}_min_odds")
        st.number_input("Puntos ATP mínimos", value=int(st.session_state[f"{prefix}_min_atp"]), step=10, key=f"{prefix}_min_atp")

    with col_ui2:
        st.markdown("##### Pesos (%)")
        st.number_input("H2H (%)", value=float(st.session_state[f"{prefix}_weight_h2h"]), step=0.5, key=f"{prefix}_weight_h2h")
        st.number_input("Reciente (%)", value=float(st.session_state[f"{prefix}_weight_recent"]), step=0.5, key=f"{prefix}_weight_recent")
        st.number_input("Superficie (%)", value=float(st.session_state[f"{prefix}_weight_surface"]), step=0.5, key=f"{prefix}_weight_surface")
        st.number_input("Ranking (%)", value=float(st.session_state[f"{prefix}_weight_ranking"]), step=0.5, key=f"{prefix}_weight_ranking")
        st.number_input("Ultra Reciente (%)", value=float(st.session_state[f"{prefix}_weight_ultra"]), step=0.5, key=f"{prefix}_weight_ultra")

    with col_ui3:
        st.markdown("##### Umbrales")
        st.number_input("Prob. Mínima para apostar (%)", value=float(st.session_state[f"{prefix}_min_prob"]), step=0.5, key=f"{prefix}_min_prob")
        if st.button("💾 Guardar Configuración", key=f"btn_save_{prefix}", use_container_width=True):
            save_analysis_config_v2(config_path, prefix)

    # --- Filtrado y Cálculos Vectorizados ---
    df = df_raw.copy()
    df = df[(df["Fecha"].dt.date >= date_start) & (df["Fecha"].dt.date <= date_end)]
    if surf_filter != "Todas":
        df = df[df["Superficie"] == surf_filter]

    if df.empty:
        st.info("No hay partidos que coincidan con los filtros de fecha/superficie.")
        return

    # Parámetros desde session_state
    w_h2h = st.session_state[f"{prefix}_weight_h2h"] / 100
    w_rec = st.session_state[f"{prefix}_weight_recent"] / 100
    w_sur = st.session_state[f"{prefix}_weight_surface"] / 100
    w_ran = st.session_state[f"{prefix}_weight_ranking"] / 100
    w_ult = st.session_state[f"{prefix}_weight_ultra"] / 100
    
    m_prob = st.session_state[f"{prefix}_min_prob"]
    m_odds = st.session_state[f"{prefix}_min_odds"]
    m_atp  = st.session_state[f"{prefix}_min_atp"]
    max_b  = st.session_state[f"{prefix}_max_bet"]
    min_b  = st.session_state[f"{prefix}_min_bet"]

    # --- Core Logica de Negocio ---
    def get_f_norm(v1, v2, weight):
        denominator = v1 + v2
        return np.where(denominator > 0, (v1 / denominator) * weight, 0)

    # Parsear columnas críticas
    h1 = df["J1 H2H %"].apply(parse_pct)
    h2 = df["J2 H2H %"].apply(parse_pct)
    r1 = df["J1 Rend. Reciente"].apply(parse_pct)
    r2 = df["J2 Rend. Reciente"].apply(parse_pct)
    s1 = df["J1 Rend. Superficie"].apply(parse_pct)
    s2 = df["J2 Rend. Superficie"].apply(parse_pct)
    u1 = df["Rend. Ultra reciente J1"].apply(parse_pct)
    u2 = df["Rend. Ultra reciente J2"].apply(parse_pct)
    p1 = pd.to_numeric(df["J1 Puntos ATP"], errors='coerce').fillna(0)
    p2 = pd.to_numeric(df["J2 Puntos ATP"], errors='coerce').fillna(0)
    o1 = pd.to_numeric(df["Cuota J1"], errors='coerce').fillna(0)
    o2 = pd.to_numeric(df["Cuota J2"], errors='coerce').fillna(0)

    # Cálculo de f1 y f2 (Modelo Estable 5 pesos)
    df["f1"] = (
        (h1/100 * w_h2h) + 
        get_f_norm(r1, r2, w_rec) + 
        get_f_norm(s1, s2, w_sur) + 
        get_f_norm(p1, p2, w_ran) + 
        get_f_norm(u1, u2, w_ult)
    ) * 100.0

    df["f2"] = (
        (h2/100 * w_h2h) + 
        get_f_norm(r2, r1, w_rec) + 
        get_f_norm(s2, s1, w_sur) + 
        get_f_norm(p2, p1, w_ran) + 
        get_f_norm(u2, u1, w_ult)
    ) * 100.0

    # Lógica de Apuesta
    def get_stake(prob_pct, odds, p1_pts, p2_pts):
        if prob_pct < m_prob: return 0, "No"
        if odds < m_odds: return 0, "No"
        # Regla: La suma de puntos ATP de los rivales debe superar el umbral
        if (p1_pts + p2_pts) < m_atp: return 0, "No"
        
        # Escala oficial del "Modelo Conservador Mejorado (15 niveles)"
        # La probabilidad aquí viene en porcentaje (0-100), por lo que dividimos entre 100 para evaluar
        p = prob_pct / 100.0
        
        if p >= 0.62:
            pct_amount = 1.0     # 100% para >= 62%
        elif p >= 0.59:
            pct_amount = 0.60
        elif p >= 0.56:
            pct_amount = 0.36
        elif p >= 0.53:
            pct_amount = 0.264
        elif p >= 0.50:
            pct_amount = 0.20
        else:
            pct_amount = 0.0

        # Calcular monto final basado en el Input de Apuesta Máxima
        final_amount = max_b * pct_amount
        
        # Restricción final: Asegurar que el monto respeta el Apuesta Mínima ingresada por UI
        if final_amount > 0 and final_amount < min_b:
            final_amount = min_b
            
        return final_amount, "Yes"

    stakes_1 = [get_stake(f, o, p1_val, p2_val) for f, o, p1_val, p2_val in zip(df["f1"], o1, p1, p2)]
    stakes_2 = [get_stake(f, o, p1_val, p2_val) for f, o, p1_val, p2_val in zip(df["f2"], o2, p1, p2)]

    bet_flags = []
    bet_for = []
    amounts = []

    for i in range(len(df)):
        s1_amt, s1_flag = stakes_1[i]
        s2_amt, s2_flag = stakes_2[i]
        
        # Si ambos cumplen, gana el de mayor probabilidad calculada
        if s1_flag == "Yes" and s2_flag == "Yes":
            if df.iloc[i]["f1"] >= df.iloc[i]["f2"]:
                s2_flag = "No"
                s2_amt = 0
            else:
                s1_flag = "No"
                s1_amt = 0
                
        if s1_flag == "Yes":
            bet_flags.append("Yes")
            bet_for.append(df.iloc[i]["Jugador 1"])
            amounts.append(s1_amt)
        elif s2_flag == "Yes":
            bet_flags.append("Yes")
            bet_for.append(df.iloc[i]["Jugador 2"])
            amounts.append(s2_amt)
        else:
            bet_flags.append("No")
            bet_for.append("")
            amounts.append(0)

    df["Bet?"] = bet_flags
    df["Bet for"] = bet_for
    df["Amount"] = amounts

    # Resultado y PNL
    def check_win(row):
        if row["Bet?"] == "No": return ""
        ganador_norm = norm(row["Ganador"])
        if not ganador_norm or ganador_norm == "-": return ""
        
        # Ignorar partidos no terminados
        if any(estado in ganador_norm for estado in ["retired", "cancelled", "walkover", "retirado", "cancelado"]):
            return ""
            
        if norm(row["Bet for"]) == ganador_norm: return "Yes"
        return "No"

    df["Won?"] = df.apply(check_win, axis=1)

    def calc_pnl(row):
        if row["Won?"] == "": return 0.0
        if row["Won?"] == "Yes":
            cuota = float(row["Cuota J1"] if norm(row["Bet for"]) == norm(row["Jugador 1"]) else row["Cuota J2"])
            return row["Amount"] * (cuota - 1)
        return -row["Amount"]

    df["PNL"] = df.apply(calc_pnl, axis=1)

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
            try:
                from scripts.refresh_data import refresh
                with st.spinner("Descargando..."):
                    refresh(mode=mode)
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

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
tab1, tab2, tab3, tab4 = st.tabs(["ATP data", "ATP Bet", "Challenger data", "Challenger Bet"])

with tab1:
    render_stats_analysis_tab("🎾 ATP History & Analysis", "atp_matches", "data/analysis_config.json", "atp", "ATP", "🔄 Refresh ATP")
with tab2:
    render_bet_tab("ATP", "atp_matches", "atp_bet", "📅 ATP Scheduled Matches")
with tab3:
    render_stats_analysis_tab("🎾 Challenger History & Analysis", "challenger_matches", "data/challenger_config.json", "cha", "CHA", "🔄 Refresh Challenger")
with tab4:
    render_bet_tab("CHA", "challenger_matches", "cha_bet", "📅 Challenger Scheduled Matches")
