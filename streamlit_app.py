import streamlit as st
import logging
import pandas as pd
from datetime import datetime
from scripts.tenis_api import (
    get_standings,
    get_tournaments,
    get_fixtures,
    get_h2h,
)
from scripts.process_files import process_fixture_period, process_fixture_surface
# TODO: Descomentar cuando streamlit-shadcn-ui se instale correctamente en Streamlit Cloud
# from streamlit_shadcn_ui import button as shadcn_button

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(page_title="🎾 Otorongo - Tennis Stats", layout="wide", page_icon="🐆")

# Load custom CSS
def load_css():
    """Load custom CSS styles"""
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        logging.warning("styles.css not found, skipping custom styles")

load_css()

# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""
    import os
    
    # Priority: Streamlit secrets > Environment variable
    # If neither is set, we allow access (default open)
    password = None
    try:
        if "APP_PASSWORD" in st.secrets:
            password = st.secrets["APP_PASSWORD"]
    except FileNotFoundError:
        pass
        
    if not password:
        password = os.getenv("APP_PASSWORD")

    # If no password configured, let them in
    if not password:
        return True

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Please enter the password to access the app", type="password", key="password_input"
    )
    
    if "password_input" in st.session_state:
        if st.session_state["password_input"] == password:
            st.session_state["password_correct"] = True
            st.rerun()
        elif st.session_state["password_input"]:
            st.error("😕 Password incorrect")

    return False

if not check_password():
    st.stop()  # Do not run the rest of the app if not authenticated
# ---------------------------

st.title("🎾🐆 Otorongo Tennis Analytics")



############################



@st.cache_data(ttl=3600)
def load_tournaments_data_v2():
    # Load from local CSV for stability and performance
    try:
        return pd.read_csv("data/tournaments.csv")
    except Exception as e:
        logging.error(f"Error loading tournaments.csv: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_standings_data(event_type):
    return get_standings(event_type=event_type, save_json=False)

@st.cache_data
def load_sample_tournaments():
    try:
        return pd.read_csv("update_20260101/sample_tournaments.csv")
    except Exception as e:
        logging.error(f"Error loading sample tournaments: {e}")
        return pd.DataFrame()

@st.dialog("Match Details")
def show_details_dialog(row):
    # CSS to increase dialog width to approx 80% of viewport
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 80vw;
            max-width: 80vw;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    view_match_details_fragment(row)

@st.fragment
def view_match_details_fragment(row):
    # Header with player selection
    col_h_1, col_h_2, col_h_3 = st.columns([2, 2, 6])
    p1_name = row['event_first_player']
    p2_name = row['event_second_player']
    
    # Use session state to track selected player for this dialog instance
    ss_key = f"details_selection_{row.get('event_key', 'unknown')}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = None

    with col_h_1:
        if st.button(f"Recent: {p1_name}", key=f"btn_p1_{row.get('event_key')}"):
             st.session_state[ss_key] = "P1"
             
    with col_h_2:
        if st.button(f"Recent: {p2_name}", key=f"btn_p2_{row.get('event_key')}"):
             st.session_state[ss_key] = "P2"
             
    with col_h_3:
         st.write(f"Details for **{p1_name}** vs **{p2_name}**")
    
    # Placeholder for API error messages
    error_placeholder = st.empty()
    api_error = False
    

def calculate_match_stats(row, df_tournaments=None, df_sample=None, df_atp_standings=None, df_wta_standings=None):
    """
    Calculates detailed statistics for a match row.
    Returns a dictionary with stats and any API error flag.
    """
    stats = {
        "Date": row.get("event_date"),
        "Player 1": row.get("event_first_player"),
        "Player 2": row.get("event_second_player"),
        "Sourface": "Unknown",
        "H2H P1": 0,
        "H2H % P1": "0%",
        "H2H P2": 0,
        "H2H % P2": "0%",
        "P1 Rec. Performance": "No data",
        "P1 Sourface R. Perf.": "No data",
        "P1 ATP Points": "No data",
        "P2 Rec. Performance": "No data",
        "P2 Sourface R. Perf.": "No data",
        "P2 ATP Points": "No data",
        "api_error": False,
        # Return full dataframes for detailed view inside dialog
        "df_p1_all": pd.DataFrame(),
        "df_p2_all": pd.DataFrame(),
        "h2h_matches_to_display": [],
        "p1_recent_text": "",
        "p2_recent_text": ""
    }
    
    # 1. Get Surface
    try:
        if df_sample is None: df_sample = load_sample_tournaments()
        if df_tournaments is None: df_tournaments = load_tournaments_data_v2()
        
        t_key = row.get("tournament_key")
        surface = "Unknown"
        
        found_in_sample = False
        if not df_sample.empty and t_key:
            match = df_sample[df_sample['tournament_key'].astype(str) == str(t_key)]
            if not match.empty:
                val = match.iloc[0].get('tournament_sourface')
                if pd.notna(val):
                    surface = val
                    found_in_sample = True

        if not found_in_sample:
            if df_tournaments is not None and not df_tournaments.empty:
                def get_surf(df, col, val):
                    match = df[df[col].astype(str) == str(val)]
                    if not match.empty:
                        c = 'tournament_sourface' if 'tournament_sourface' in match.columns else 'tournament_surface'
                        return match.iloc[0][c]
                    return None

                if t_key:
                    res = get_surf(df_tournaments, 'tournament_key', t_key)
                    if res: surface = res
                
                if surface == "Unknown" and row.get("tournament_name"):
                    res = get_surf(df_tournaments, 'tournament_name', row['tournament_name'])
                    if res: surface = res
        
        # Normalize Surface to match grouping logic
        s_lower = str(surface).lower()
        if "hard" in s_lower:
            surface = "Hard"
        elif "clay" in s_lower:
            surface = "Clay"
        elif "grass" in s_lower:
            surface = "Grass"
            
        stats["Sourface"] = surface
    except Exception as e:
        logging.error(f"Error fetching surface: {e}")

    # 2. Get H2H
    try:
        p1_key = row.get("first_player_key")
        p2_key = row.get("second_player_key")
        
        if p1_key and p2_key:
            h2h_data = get_h2h(first_player_key=p1_key, second_player_key=p2_key, save_json=False, save_csv=False)
            
            if h2h_data is None:
                stats["api_error"] = True
            elif "H2H" in h2h_data:
                h2h_list = h2h_data["H2H"]
                total_matches = len(h2h_list)
                
                if total_matches > 0:
                    h2h_p1 = 0
                    h2h_p2 = 0
                    for match in h2h_list:
                        winner = match.get("event_winner")
                        match_p1 = match.get("first_player_key")
                        match_p2 = match.get("second_player_key")
                        
                        winner_key = None
                        if winner == "First Player":
                            winner_key = match_p1
                        elif winner == "Second Player":
                            winner_key = match_p2
                        else:
                            winner_key = winner
                            
                        clean_winner = str(int(float(winner_key))) if winner_key else None
                        clean_p1 = str(int(float(p1_key))) if p1_key else None
                        clean_p2 = str(int(float(p2_key))) if p2_key else None
                        
                        if clean_winner and clean_p1 and clean_winner == clean_p1:
                            h2h_p1 += 1
                        elif clean_winner and clean_p2 and clean_winner == clean_p2:
                            h2h_p2 += 1
                    
                    stats["H2H P1"] = h2h_p1
                    stats["H2H % P1"] = f"{(h2h_p1 / total_matches) * 100:.1f}%".replace('.', ',')
                    stats["H2H P2"] = h2h_p2
                    stats["H2H % P2"] = f"{(h2h_p2 / total_matches) * 100:.1f}%".replace('.', ',')

                    # Last 5 matches
                    h2h_display = []
                    for match in h2h_list[:5]:
                        winner_name = match.get("event_winner", "Unknown")
                        clean_winner = str(int(float(match.get("event_winner")))) if match.get("event_winner") and str(match.get("event_winner")).replace('.','',1).isdigit() else match.get("event_winner")
                        
                        display_winner = winner_name
                        p1_name = row.get("event_first_player")
                        p2_name = row.get("event_second_player")
                        
                        clean_p1 = str(int(float(p1_key))) if p1_key else None
                        clean_p2 = str(int(float(p2_key))) if p2_key else None
                        
                        if clean_winner == "First Player" or clean_winner == clean_p1:
                            display_winner = p1_name
                        elif clean_winner == "Second Player" or clean_winner == clean_p2:
                            display_winner = p2_name
                            
                        h2h_display.append({
                            "Date": match.get("event_date"),
                            "Tournament": match.get("tournament_name"),
                            "Winner": display_winner,
                            "Score": match.get("event_final_result")
                        })
                    stats["h2h_matches_to_display"] = h2h_display

    except Exception as e:
        logging.error(f"Error fetching H2H: {e}")

    # 3. Get ATP/WTA Points
    try:
        def find_points(df, p_key):
             if df is not None and not df.empty and 'player_key' in df.columns and 'points' in df.columns:
                 try:
                     target_key = float(p_key)
                     match = df[df['player_key'].apply(lambda x: float(x) if pd.notnull(x) else -1) == target_key]
                     if not match.empty:
                         return match.iloc[0]['points']
                 except:
                     return None
             return None

        if df_atp_standings is None: df_atp_standings = load_standings_data("ATP")
        
        p1_key = row.get("first_player_key")
        p2_key = row.get("second_player_key")

        # P1 Points
        pt = find_points(df_atp_standings, p1_key)
        if pt is None:
            if df_wta_standings is None: df_wta_standings = load_standings_data("WTA")
            pt = find_points(df_wta_standings, p1_key)
        if pt is not None: stats["P1 ATP Points"] = pt

        # P2 Points  
        pt = find_points(df_atp_standings, p2_key)
        if pt is None:
            if df_wta_standings is None: df_wta_standings = load_standings_data("WTA")
            pt = find_points(df_wta_standings, p2_key)
        if pt is not None: stats["P2 ATP Points"] = pt

    except Exception as e:
        logging.error(f"Error fetching standings: {e}")

    # 4. Recent Performance
    recent_start = (datetime.now() - pd.DateOffset(days=365)).strftime("%Y-%m-%d")
    recent_end = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # P1
        if p1_key:
             clean_p1_key = str(int(float(p1_key))) if str(p1_key).replace('.','',1).isdigit() else str(p1_key)
             df_p1 = get_fixtures(date_start=recent_start, date_stop=recent_end, player_key=clean_p1_key, save_json=False, save_csv=False)
             
             if df_p1 is None:
                 stats["api_error"] = True
             elif not df_p1.empty:
                 # Deduplicate by event_key to ensure consistency
                 if 'event_key' in df_p1.columns:
                     df_p1 = df_p1.drop_duplicates(subset=['event_key'])

                 if 'event_type_type' in df_p1.columns:
                     df_p1 = df_p1[df_p1['event_type_type'].astype(str).str.contains("Singles", case=False, na=False)]
                 
                 stats["df_p1_all"] = df_p1.copy()
                 
                 stats_p1 = process_fixture_period(df_p1, save_csv=False)
                 if not stats_p1.empty:
                      row_p1 = stats_p1[stats_p1['results_for_player_key'] == clean_p1_key]
                      if not row_p1.empty:
                          w = row_p1.iloc[0]['won_main_player']
                          l = row_p1.iloc[0]['lost_main_player']
                          total = w + l
                          pct = (w / total * 100) if total > 0 else 0
                          stats["P1 Rec. Performance"] = f"{pct:.1f}%".replace('.', ',')
                          stats["p1_recent_text"] = f"{row.get('event_first_player')}: {w}W - {l}L"
                      
                 stats_p1_surf = process_fixture_surface(df_p1, save_csv=False)
                 if not stats_p1_surf.empty and stats["Sourface"] != "Unknown":
                      row_surf = stats_p1_surf[
                          (stats_p1_surf['results_for_player_key'] == clean_p1_key) & 
                          (stats_p1_surf['tournament_sourface'] == stats["Sourface"])
                      ]
                      if not row_surf.empty:
                           w = row_surf.iloc[0]['won_main_player']
                           l = row_surf.iloc[0]['lost_main_player']
                           total = w + l
                           pct = (w / total * 100) if total > 0 else 0
                           stats["P1 Sourface R. Perf."] = f"{pct:.1f}%".replace('.', ',')
                           stats["p1_recent_text"] += f" | Sourface: {w}W - {l}L"

        # P2
        if p2_key:
             clean_p2_key = str(int(float(p2_key))) if str(p2_key).replace('.','',1).isdigit() else str(p2_key)
             df_p2 = get_fixtures(date_start=recent_start, date_stop=recent_end, player_key=clean_p2_key, save_json=False, save_csv=False)
             
             if df_p2 is None:
                 stats["api_error"] = True
             elif not df_p2.empty:
                 # Deduplicate
                 if 'event_key' in df_p2.columns:
                     df_p2 = df_p2.drop_duplicates(subset=['event_key'])

                 if 'event_type_type' in df_p2.columns:
                     df_p2 = df_p2[df_p2['event_type_type'].astype(str).str.contains("Singles", case=False, na=False)]

                 stats["df_p2_all"] = df_p2.copy()

                 stats_p2 = process_fixture_period(df_p2, save_csv=False)
                 if not stats_p2.empty:
                      row_p2 = stats_p2[stats_p2['results_for_player_key'] == clean_p2_key]
                      if not row_p2.empty:
                          w = row_p2.iloc[0]['won_main_player']
                          l = row_p2.iloc[0]['lost_main_player']
                          total = w + l
                          pct = (w / total * 100) if total > 0 else 0
                          stats["P2 Rec. Performance"] = f"{pct:.1f}%".replace('.', ',')
                          stats["p2_recent_text"] = f"{row.get('event_second_player')}: {w}W - {l}L"
                 
                 stats_p2_surf = process_fixture_surface(df_p2, save_csv=False)
                 if not stats_p2_surf.empty and stats["Sourface"] != "Unknown":
                      row_surf = stats_p2_surf[
                           (stats_p2_surf['results_for_player_key'] == clean_p2_key) & 
                           (stats_p2_surf['tournament_sourface'] == stats["Sourface"])
                      ]
                      if not row_surf.empty:
                           w = row_surf.iloc[0]['won_main_player']
                           l = row_surf.iloc[0]['lost_main_player']
                           total = w + l
                           pct = (w / total * 100) if total > 0 else 0
                           stats["P2 Sourface R. Perf."] = f"{pct:.1f}%".replace('.', ',')
                           stats["p2_recent_text"] += f" | Sourface: {w}W - {l}L"

    except Exception as e:
        logging.error(f"Error fetching recent performance: {e}")
        
    return stats

@st.dialog("Match Details")
def show_details_dialog(row):
    # CSS to increase dialog width to approx 80% of viewport
    st.markdown(
        """
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {
            width: 80vw;
            max-width: 80vw;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    view_match_details_fragment(row)

@st.fragment
def view_match_details_fragment(row):
    # Header with player selection
    col_h_1, col_h_2, col_h_3 = st.columns([2, 2, 6])
    p1_name = row['event_first_player']
    p2_name = row['event_second_player']
    
    # Use session state to track selected player for this dialog instance
    ss_key = f"details_selection_{row.get('event_key', 'unknown')}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = None

    with col_h_1:
        if st.button(f"Recent: {p1_name}", key=f"btn_p1_{row.get('event_key')}", use_container_width=True):
             st.session_state[ss_key] = "P1"
             
    with col_h_2:
        if st.button(f"Recent: {p2_name}", key=f"btn_p2_{row.get('event_key')}", use_container_width=True):
             st.session_state[ss_key] = "P2"
             
    with col_h_3:
         st.write(f"Details for **{p1_name}** vs **{p2_name}**")
    
    # Placeholder for API error messages
    error_placeholder = st.empty()

    # Calculate stats using the helper function
    # Note: we might want to cache this per session/fragment if possible, but fragments rerun on interaction
    # For now, let's call it. If it's too slow, we can cache.
    # The helper functions (get_fixtures etc) are cached, so it shouldn't be too bad.
    stats = calculate_match_stats(row)

    # 4. Build DataFrame for Display
    # Extract keys to match DataFrame columns
    display_keys = [
        "Date", "Player 1", "Player 2", "Sourface", 
        "H2H P1", "H2H % P1", "H2H P2", "H2H % P2", 
        "P1 Rec. Performance", "P1 Sourface R. Perf.", "P1 ATP Points", 
        "P2 Rec. Performance", "P2 Sourface R. Perf.", "P2 ATP Points"
    ]
    
    # Filter stats dict to just these keys
    data = {k: [stats.get(k)] for k in display_keys}
    df_details = pd.DataFrame(data)
    
    # Display
    st.dataframe(df_details, hide_index=True, use_container_width=True)
    
    # Recent Performance Text
    if stats.get("p1_recent_text"):
        st.caption(stats.get("p1_recent_text"))
    if stats.get("p2_recent_text"):
        st.caption(stats.get("p2_recent_text"))

    # H2H History Table
    h2h_matches_to_display = stats.get("h2h_matches_to_display", [])
    if h2h_matches_to_display:
        st.markdown("##### Head-to-Head History (Last 5)")
        st.dataframe(pd.DataFrame(h2h_matches_to_display), hide_index=True, use_container_width=True)
    
    # Recent Matches Section
    selected_player_side = st.session_state.get(f"details_selection_{row.get('event_key', 'unknown')}")
    
    if selected_player_side:
        target_name = row['event_first_player'] if selected_player_side == "P1" else row['event_second_player']
        target_df = stats.get("df_p1_all") if selected_player_side == "P1" else stats.get("df_p2_all")
        
        st.markdown(f"---")
        st.subheader(f"Recent matches of {target_name}")
        
        # Helper to format table
        def get_recent_matches_display(df_matches, target_surface=None):
            if df_matches is None or df_matches.empty:
                return pd.DataFrame()
            
            df = df_matches.copy()
            
            # Merge surface if needed
            # We always try to merge if 'tournament_sourface' is missing OR if we need to filter by it
            if 'tournament_sourface' not in df.columns or target_surface:
                 df_t = load_tournaments_data_v2()
                 
                 if df_t is not None and not df_t.empty and 'tournament_key' in df.columns:
                     try:
                         # Ensure we have the surface column in tournaments df
                         surf_col = 'tournament_sourface'
                         if 'tournament_sourface' not in df_t.columns and 'tournament_surface' in df_t.columns:
                             df_t['tournament_sourface'] = df_t['tournament_surface']
                         
                         if surf_col in df_t.columns:
                             # Prepare match df for merge
                             df['t_key_str'] = df['tournament_key'].astype(str).str.split('.').str[0]
                             
                             # Prepare tournaments df for merge
                             df_t_merge = df_t.copy()
                             df_t_merge['t_key_str'] = df_t_merge['tournament_key'].astype(str).str.split('.').str[0]
                             
                             # Deduplicate to avoid exploding rows
                             df_t_merge = df_t_merge[['t_key_str', surf_col]].drop_duplicates(subset=['t_key_str'])
                             
                             # Drop existing surface col in matches if present to avoid suffixes
                             if surf_col in df.columns:
                                 df = df.drop(columns=[surf_col])
                                 
                             # Merge
                             df = df.merge(df_t_merge, on='t_key_str', how='left')
                     except Exception as e:
                         logging.error(f"Error merging surface info: {e}")

            # Filter by surface if requested
            if target_surface:
                 if 'tournament_sourface' in df.columns:
                     # Filter: handle potential NaNs
                     df = df[df['tournament_sourface'].astype(str) == str(target_surface)]
                 else:
                     return pd.DataFrame() # Cannot filter if column missing

            if df.empty:
                return pd.DataFrame()
            
            # Resolve winner name
            def resolve_winner(r):
                w = str(r.get('event_winner', ''))
                p1_k = str(r.get('first_player_key', '')).split('.')[0]
                p2_k = str(r.get('second_player_key', '')).split('.')[0]
                
                # If winner is 'First Player'
                if w == "First Player": return r.get('event_first_player')
                if w == "Second Player": return r.get('event_second_player')
                
                # If winner is key
                w_clean = w.split('.')[0]
                if w_clean == p1_k: return r.get('event_first_player')
                if w_clean == p2_k: return r.get('event_second_player')
                
                return w

            df['Winner_Name'] = df.apply(resolve_winner, axis=1)
            
            # Renaming and Selection
            df = df.reset_index(drop=True)
            df.index += 1
            df['#'] = df.index
            
            cols_map = {
                'event_date': 'Date',
                'tournament_name': 'Tournament',
                'event_first_player': 'P1',
                'event_second_player': 'P2',
                'Winner_Name': 'Winner',
                'event_final_result': 'Score'
            }
            
            # Ensure columns exist before selecting
            defaults = {k: '' for k in cols_map.keys()}
            for k in defaults:
                if k not in df.columns:
                    df[k] = defaults[k]
                    
            final_df = df[['#'] + list(cols_map.keys())].rename(columns=cols_map)
            return final_df

        st.markdown("**Recent Matches**")
        if not target_df.empty:
             df_disp_1 = get_recent_matches_display(target_df)
             st.dataframe(df_disp_1, hide_index=True, use_container_width=True)
        else:
             st.info("No recent matches found.")
             
        surface = stats.get("Sourface", "Unknown")
        st.markdown(f"**Recent Matches in Sourface ({surface})**")
        if surface != "Unknown" and not target_df.empty:
             df_disp_2 = get_recent_matches_display(target_df, target_surface=surface)
             if not df_disp_2.empty:
                 st.dataframe(df_disp_2, hide_index=True, use_container_width=True)
             else:
                 st.info(f"No recent matches on {surface}.")
        elif surface == "Unknown":
             st.warning("Current match surface is unknown, cannot filter.")
        else:
             st.info(f"No recent matches on {surface}.")

    # Copy functionality (Client-side friendly)
    st.markdown("##### Export Data")
    col_copy1, col_copy2 = st.columns([1, 1])
    
    csv_string = df_details.to_csv(sep='\t', index=False, header=False)
    
    with col_copy1:
        st.download_button(
            label="📥 Download for Excel (.tsv)",
            data=csv_string,
            file_name=f"match_details_{row.get('event_date', 'date')}.tsv",
            mime="text/tab-separated-values",
        )
    
    with col_copy2:
        st.code(csv_string, language="text")
        st.caption("☝️ Click the copy icon in the top right.")

    # Display API Error if flagged
    if stats.get("api_error"):
        with error_placeholder.container():
            st.error("API is currently unstable. Please try again in a few moments.")
            if st.button("🔄 Refresh Results"):
                st.rerun()



# ----------------------
# Search Events Section
# ----------------------
tab1, tab2 = st.tabs(["Search Events", "Day Report"])

with tab1:
    st.markdown("### 🔍 Search Events")

    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    with col1:
        search_date = st.date_input(
            "Select Date", 
            value=datetime.today(),
            key="search_events_date"
        )
    with col2:
        league_filter = st.selectbox("League", ["All", "ATP", "WTA", "Mixed"], key="search_events_league")
    with col3:
        format_filter = st.selectbox("Format", ["All", "Singles", "Doubles"], key="search_events_format")
        
    with col4:
        st.write("") # Spacer for better vertical alignment
        st.write("") 
        search_clicked = shadcn_button("Search", key="search_events_btn", variant="default")

    # Initialize session state variable to store results if not present
    if "search_events_results" not in st.session_state:
        st.session_state.search_events_results = None

    # If search button is clicked, fetch data and update session state
    if search_clicked:
        with st.spinner("Fetching matches..."):
            try:
                # Reuse get_fixtures to fetch data for the single selected date
                df_search = get_fixtures(
                    date_start=search_date.strftime("%Y-%m-%d"),
                    date_stop=search_date.strftime("%Y-%m-%d"),
                    save_json=False
                )
                st.session_state.search_events_results = df_search
            except Exception as e:
                st.error(f"❌ Error fetching events: {str(e)}")
                st.session_state.search_events_results = None

    # Display results if available in session state
    if st.session_state.search_events_results is not None:
        df_search = st.session_state.search_events_results.copy()
        
        # Apply filters
        if not df_search.empty:
            if league_filter != "All":
                if league_filter == "Mixed":
                    df_search = df_search[df_search['event_type_type'].str.contains("Mix", case=False, na=False)]
                else:
                    df_search = df_search[df_search['event_type_type'].str.contains(league_filter, case=False, na=False)]
                    
            if format_filter != "All":
                df_search = df_search[df_search['event_type_type'].str.contains(format_filter, case=False, na=False)]
        
        if not df_search.empty:
            st.success(f"Found {len(df_search)} matches for {search_date.strftime('%Y-%m-%d')}")
            
            # Filter by Tournament
            tournaments = sorted(df_search['tournament_name'].dropna().unique().tolist())
            selected_tournament = st.selectbox("Filter by Tournament", ["All"] + tournaments, key="search_events_tournament")
            
            if selected_tournament != "All":
                df_search = df_search[df_search['tournament_name'] == selected_tournament]
            
            # Header row
            h1, h2, h3, h4, h5 = st.columns([1, 2, 3, 2, 2])
            h1.markdown("**Time**")
            h2.markdown("**Tournament**")
            h3.markdown("**Match**")
            h4.markdown("**Result**")
            h5.markdown("**Statistics**")
            
            st.divider()

            for index, row in df_search.iterrows():
                c1, c2, c3, c4, c5 = st.columns([1, 2, 3, 2, 2])
                
                # Time
                c1.write(f"{row.get('event_time', 'N/A')}")
                
                # Tournament
                c2.write(f"{row.get('tournament_name', 'N/A')}")
                
                # Match (Players)
                p1 = row.get('event_first_player', 'Player 1')
                p2 = row.get('event_second_player', 'Player 2')
                c3.write(f"{p1} vs {p2}")
                
                # Result
                res = row.get('event_final_result', '-')
                c4.write(res)
                
                # Action Button
                # Use a unique key for each button depending on event_key
                with c5:
                    if shadcn_button("See Details", key=f"btn_details_{row.get('event_key', index)}", variant="default"):
                        show_details_dialog(row)
                    
                # Add a visual separator
                st.markdown("---")

        else:
            st.info("No matches scheduled for this date.")

with tab2:
    st.header("Day Report")
    
    # Use vertical_alignment="bottom" to align input fields and button
    col_dr_1, col_dr_2, col_dr_3 = st.columns([2, 2, 2], vertical_alignment="bottom")
    with col_dr_1:
         dr_date = st.date_input("Select Date", value=datetime.today(), key="dr_date")
    with col_dr_2:
         dr_format = st.selectbox("Format", ["All", "Singles", "Doubles"], key="dr_format")
    with col_dr_3:
         dr_see_tournaments = st.button("See Tournaments", type="secondary")
    
    # Session state for tournament list
    if "dr_tournaments_list" not in st.session_state:
        st.session_state.dr_tournaments_list = []
        st.session_state.dr_fixtures_cache = None
    
    # Track previous date and format to detect changes
    if "dr_prev_date" not in st.session_state:
        st.session_state.dr_prev_date = None
    if "dr_prev_format" not in st.session_state:
        st.session_state.dr_prev_format = None
    
    # Reset results if date or format changed
    current_date_str = dr_date.strftime("%Y-%m-%d")
    if (st.session_state.dr_prev_date != current_date_str or 
        st.session_state.dr_prev_format != dr_format):
        # Clear previous results
        st.session_state.dr_show_results = False
        st.session_state.dr_current_tournament = None
        st.session_state.dr_tournaments_list = []
        st.session_state.dr_fixtures_cache = None
        # Update tracking
        st.session_state.dr_prev_date = current_date_str
        st.session_state.dr_prev_format = dr_format

    if dr_see_tournaments:
        with st.spinner("Fetching tournaments for date..."):
             dates_str = dr_date.strftime("%Y-%m-%d")
             df_fix = get_fixtures(date_start=dates_str, date_stop=dates_str, save_json=False, save_csv=False)
             
             if df_fix is not None and not df_fix.empty:
                  # Filter format
                  if dr_format != "All":
                      df_fix = df_fix[df_fix['event_type_type'].astype(str).str.contains(dr_format, case=False, na=False)]
                  
                  if not df_fix.empty:
                      t_list = sorted(df_fix['tournament_name'].dropna().unique().tolist())
                      st.session_state.dr_tournaments_list = t_list
                      st.session_state.dr_fixtures_cache = df_fix
                  else:
                      st.warning(f"No {dr_format} matches found for this date.")
                      st.session_state.dr_tournaments_list = []
                      st.session_state.dr_fixtures_cache = None
             else:
                  st.warning("No matches found for this date.")
                  st.session_state.dr_tournaments_list = []
                  st.session_state.dr_fixtures_cache = None
    
    # 2nd Row: Tournament Select & Go
    if st.session_state.dr_tournaments_list:
        st.markdown("---")
        # Align Go button with Selectbox
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        with c1:
             dr_selected_tournament = st.selectbox("Select Tournament", st.session_state.dr_tournaments_list, key="dr_selected_t", index=None, placeholder="Choose a tournament...")
        with c2:
             dr_go = st.button("Go", type="primary")
             
        if dr_go and dr_selected_tournament:
             st.session_state.dr_show_results = True
             st.session_state.dr_current_tournament = dr_selected_tournament
        
        # Check if we should show results: only if Go was clicked AND selection matches
        if (st.session_state.get("dr_show_results") and 
            st.session_state.get("dr_current_tournament") and
            dr_selected_tournament == st.session_state.dr_current_tournament):
             # Use the stored tournament from session state to remain consistent
             current_tournament = st.session_state.dr_current_tournament
             
             df_cache = st.session_state.dr_fixtures_cache
             if df_cache is not None:
                  # Filter specific tournament
                  df_t_matches = df_cache[df_cache['tournament_name'] == current_tournament]
                  
                  if not df_t_matches.empty:
                       st.info(f"Processing {len(df_t_matches)} matches for {current_tournament}...")
                       
                       # Process matches
                       # We don't want to show progress bar on every interaction (like opening dialog), 
                       # but it's fine for now or could be guarded.
                       progress_bar = st.progress(0)
                       total = len(df_t_matches)
                       
                       for idx, (i, row) in enumerate(df_t_matches.iterrows()):
                            # Calculate stats
                            stats = calculate_match_stats(row)
                            
                            # Construct 1-row DF for display
                            display_keys = [
                                "Date", "Player 1", "Player 2", "Sourface", 
                                "H2H P1", "H2H % P1", "H2H P2", "H2H % P2", 
                                "P1 Rec. Performance", "P1 Sourface R. Perf.", "P1 ATP Points", 
                                "P2 Rec. Performance", "P2 Sourface R. Perf.", "P2 ATP Points"
                            ]
                            data = {k: [stats.get(k)] for k in display_keys}
                            df_mini = pd.DataFrame(data)
                            
                            # Display row
                            # Determine Status Color
                            status = row.get('event_status', '')
                            # API Statuses: "Finished", "Cancelled" (Red)
                            # "After Pending", "" (Green)
                            # "Ended", "Awarded", "Retired" (Red)
                            # "Int.", "Postp." (Red?)
                            # Live statuses: "1. Set", "2. Set", "3. Set", "4. Set", "5. Set", "Points", "In Progress" (Orange)
                            
                            status_color = "green"
                            status_label = "Not Started"
                            
                            # Normalize status
                            s_lower = str(status).lower().strip()
                            
                            if s_lower in ["finished", "ended", "retired", "awarded", "cancelled", "walkover", "after retired"]:
                                status_color = "red"
                                status_label = "Finished"
                                if s_lower == "cancelled": status_label = "Cancelled"
                                if s_lower == "retired": status_label = "Retired"
                            elif s_lower in ["", "after pending", "postponed"]:
                                status_color = "green"
                                status_label = "Not Started"
                                if s_lower == "postponed": status_label = "Postponed"
                            elif row.get("event_live") == "1" or "set" in s_lower or "game" in s_lower or "point" in s_lower or "progress" in s_lower:
                                status_color = "orange"
                                status_label = f"Ongoing ({status})"
                            else:
                                status_color = "green" 
                                status_label = "Not Started"
                            
                            # Determine Winner if Finished
                            winner_text = ""
                            if status_label == "Finished" or status_label == "Retired" or status_label == "Walkover":
                                w = str(row.get("event_winner", ""))
                                w_name = ""
                                if w == "First Player":
                                    w_name = row.get("event_first_player")
                                elif w == "Second Player":
                                    w_name = row.get("event_second_player")
                                else:
                                    w_name = "" 
                                
                                if w_name:
                                    winner_text = f" | **Winner: {w_name}**"

                            # Match Header Layout
                            # Using vertical-alignment friendly layout if possible, but columns is standard
                            # Adjust ratios: Title (approx 45%), Button (approx 5%), Status (approx 50%)
                            c_head_1, c_head_2, c_head_3 = st.columns([0.45, 0.05, 0.50], vertical_alignment="center")
                            
                            with c_head_1:
                                st.markdown(f"**Match {idx+1}: {row.get('event_first_player')} vs {row.get('event_second_player')}** ({row.get('event_time')})")
                            
                            with c_head_2:
                                # Small button with unique key
                                if st.button("🔍", key=f"dr_details_{idx}_{row.get('event_key')}", help="Open Match Details"):
                                     show_details_dialog(row)
                            
                            with c_head_3:
                                st.markdown(f":{status_color}[● {status_label}]{winner_text}")

                            col_table, col_copy = st.columns([8, 1], vertical_alignment="center")
                            with col_table:
                                 st.dataframe(df_mini, hide_index=True, use_container_width=True)
                            
                            with col_copy:
                                 # Prepare copy text (simplified for CSV/Excel paste)
                                 # Format: Date \t P1 \t P2 ...
                                 copy_text_val = "\t".join([str(stats.get(k, "")) for k in display_keys])
                                 st.code(copy_text_val, language="text")
                            
                            progress_bar.progress((idx + 1) / total)
                       
                       progress_bar.empty()
                  else:
                       st.warning("No matches found for this tournament.")
