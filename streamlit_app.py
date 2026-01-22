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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(page_title="Tennis API Data Fetcher", layout="wide")

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

st.title("🎾 Tennis API Data Fetcher")



############################
st.subheader("Search player ID")


@st.cache_data
def load_players():
    df = pd.read_csv("data/player_ids_fixtures.csv")
    return df


players_df = load_players()

search_term = st.text_input("Search player name:")

if search_term:
    # Filter players matching the search
    filtered = players_df[
        players_df["player_name"].str.contains(search_term, case=False, na=False)
    ]

    if not filtered.empty:
        selected = st.selectbox("Select player:", filtered["player_name"].tolist())
        if selected:
            player_key = filtered[filtered["player_name"] == selected][
                "player_key"
            ].values[0]
            st.write(f"Player ID: {player_key}")
    else:
        st.warning("No players found")

st.markdown("---")
############################


@st.cache_data(ttl=3600)
def load_tournaments_data_v2():
    return get_tournaments(save_json=False)

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
    
    # 1. Get Surface
    surface = "Unknown"
    try:
        # Load sample tournaments first as per requirement
        df_sample = load_sample_tournaments()
        t_key = row.get("tournament_key")
        
        found_in_sample = False
        if not df_sample.empty and t_key:
            # clean key for comparison
            # t_key from row might be int or str, csv usually has int or str
            # let's try strict matching first, then types
            match = df_sample[df_sample['tournament_key'].astype(str) == str(t_key)]
            if not match.empty:
                if 'tournament_sourface' in match.columns:
                    val = match.iloc[0]['tournament_sourface']
                    if pd.notna(val):
                        surface = val
                        found_in_sample = True

        # Fallback to API/old method if not found in sample csv
        if not found_in_sample:
            df_tournaments = load_tournaments_data_v2()
            if df_tournaments is not None and not df_tournaments.empty:
                # Robust matching helper
                def get_surf(df, col, val):
                    # Try string matching for keys/names
                    match = df[df[col].astype(str) == str(val)]
                    if not match.empty:
                        # 'tournament_surface' vs 'tournament_sourface'
                        c = 'tournament_sourface' if 'tournament_sourface' in match.columns else 'tournament_surface'
                        return match.iloc[0][c]
                    return None

                if t_key:
                    res = get_surf(df_tournaments, 'tournament_key', t_key)
                    if res: surface = res
                
                # Fallback to name if key didn't work or wasn't present
                if surface == "Unknown" and row.get("tournament_name"):
                    res = get_surf(df_tournaments, 'tournament_name', row['tournament_name'])
                    if res: surface = res
    except Exception as e:
        logging.error(f"Error fetching surface: {e}")

    # 2. Get H2H
    h2h_p1 = 0
    h2h_p2 = 0
    h2h_p1_pct = "0%"
    h2h_p2_pct = "0%"
    h2h_matches_to_display = []
    
    try:
        p1_key = row.get("first_player_key")
        p2_key = row.get("second_player_key")
        
        if p1_key and p2_key:
            # Fetch H2H data
            h2h_data = get_h2h(first_player_key=p1_key, second_player_key=p2_key, save_json=False, save_csv=False)
            
            if h2h_data is None:
                api_error = True
            elif "H2H" in h2h_data:
                h2h_list = h2h_data["H2H"]
                total_matches = len(h2h_list)
                
            if total_matches > 0:
                    # Count wins
                    # API returns "First Player" or "Second Player" in 'event_winner'
                    # Or sometimes the key directly? We handle both.
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
                            
                        # Clean keys for safe comparison
                        clean_winner = str(int(float(winner_key))) if winner_key else None
                        clean_p1 = str(int(float(p1_key))) if p1_key else None
                        clean_p2 = str(int(float(p2_key))) if p2_key else None
                        
                        if clean_winner and clean_p1 and clean_winner == clean_p1:
                            h2h_p1 += 1
                        elif clean_winner and clean_p2 and clean_winner == clean_p2:
                            h2h_p2 += 1
                    
                    
                    h2h_p1_pct = f"{(h2h_p1 / total_matches) * 100:.1f}%".replace('.', ',')
                    h2h_p2_pct = f"{(h2h_p2 / total_matches) * 100:.1f}%".replace('.', ',')
                    
                    # Process last 5 matches for display
                    # API sorts by date desc usually? Let's take first 5
                    for match in h2h_list[:5]:
                        winner_name = match.get("event_winner", "Unknown")
                        # If winner is "First Player" mapped to name? 
                        # API usually returns index or "First Player", but let's try to map to names if possible
                        # Actually 'event_winner' matches P1/P2 keys logic we just did
                        
                        clean_winner = str(int(float(match.get("event_winner")))) if match.get("event_winner") and match.get("event_winner").replace('.','',1).isdigit() else match.get("event_winner")
                        
                        display_winner = winner_name
                        p1_name = row.get("event_first_player")
                        p2_name = row.get("event_second_player")
                        
                        clean_p1 = str(int(float(p1_key))) if p1_key else None
                        clean_p2 = str(int(float(p2_key))) if p2_key else None
                        
                        if clean_winner == "First Player" or clean_winner == clean_p1:
                            display_winner = p1_name
                        elif clean_winner == "Second Player" or clean_winner == clean_p2:
                            display_winner = p2_name
                            
                        h2h_matches_to_display.append({
                            "Date": match.get("event_date"),
                            "Tournament": match.get("tournament_name"),
                            "Winner": display_winner,
                            "Score": match.get("event_final_result")
                        })
                    
    except Exception as e:
        logging.error(f"Error fetching H2H: {e}")

    # 3. Get ATP Points (Standings)
    p1_points = "No data"
    p2_points = "No data"
    
    try:
        # Determine event type - simplistic heuristic or try ATP then WTA
        # This is strictly for demonstration; robust gender detection needs more data
        # fixtures data might include 'event_type_type' but that might be 'Singles' etc. 
        # API documentation says get_standings(event_type='ATP'|'WTA'|'ITF')
        
        # We will attempt to check both ATP and WTA if not sure, or prioritize based on tournament name?
        # Let's try fetching ATP first.
        
        # Helper to find points
        def find_points(df, p_key):
             if df is not None and not df.empty and 'player_key' in df.columns and 'points' in df.columns:
                 try:
                     # Robust comparison dealing with int vs float vs str types
                     # Convert both series and key to float for comparison
                     target_key = float(p_key)
                     match = df[df['player_key'].apply(lambda x: float(x) if pd.notnull(x) else -1) == target_key]
                     
                     if not match.empty:
                         return match.iloc[0]['points']
                 except Exception as err:
                     logging.error(f"Error matching points for key {p_key}: {err}")
                     return None
             return None

        # Fetch ATP
        df_atp = load_standings_data("ATP")
        p1_pt = find_points(df_atp, p1_key)
        if p1_pt is not None:
             p1_points = p1_pt
        
        p2_pt = find_points(df_atp, p2_key)
        if p2_pt is not None:
             p2_points = p2_pt
             
        # If still no data, maybe it's WTA?
        if p1_points == "No data" or p2_points == "No data":
             df_wta = load_standings_data("WTA")
             
             if p1_points == "No data":
                 val = find_points(df_wta, p1_key)
                 if val is not None: p1_points = val
                 
             if p2_points == "No data":
                 val = find_points(df_wta, p2_key)
                 if val is not None: p2_points = val

    except Exception as e:
        logging.error(f"Error fetching standings: {e}")

    # 4. Recent Performance
    p1_recent = "No data"
    p1_recent_text = ""
    p1_surface_recent = "No data"
    p2_recent = "No data"
    p2_recent_text = ""
    p2_surface_recent = "No data"
    
    # Store full dataframes for detailed tables
    df_p1_all = pd.DataFrame()
    df_p2_all = pd.DataFrame()
    
    # Define recent period (e.g. last 6 months)
    recent_start = (datetime.now() - pd.DateOffset(days=365)).strftime("%Y-%m-%d")
    recent_end = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # P1
        if p1_key:
             # Clean key to ensure int-string format "1099"
             try:
                 clean_p1_key = str(int(float(p1_key)))
             except:
                 clean_p1_key = str(p1_key)

             # get_fixtures now returns a dataframe!
             df_p1 = get_fixtures(date_start=recent_start, date_stop=recent_end, player_key=clean_p1_key, save_json=False, save_csv=False)
             
             if df_p1 is None:
                 api_error = True
             elif not df_p1.empty:
                 # Filter for Singles only
                 if 'event_type_type' in df_p1.columns:
                     df_p1 = df_p1[df_p1['event_type_type'].astype(str).str.contains("Singles", case=False, na=False)]
                 
                 # Capture for detailed view
                 df_p1_all = df_p1.copy()
                 
                 # Process overall
                 # We use a temp filename to avoid overwriting main data files if we care, or just overwrite.
                 # Using /dev/null or similar might be better but let's just use a temp name.
                 stats_p1 = process_fixture_period(df_p1, save_csv=False)
                 if not stats_p1.empty:
                      # Clean key for matching (handle float string '1234.0' -> '1234')
                      target_key = clean_p1_key
                      row_p1 = stats_p1[stats_p1['results_for_player_key'] == target_key]
                      if not row_p1.empty:
                          w = row_p1.iloc[0]['won_main_player']
                          l = row_p1.iloc[0]['lost_main_player']
                          total = w + l
                          pct = (w / total * 100) if total > 0 else 0
                          p1_recent = f"{pct:.1f}%".replace('.', ',')
                          p1_recent_text = f"{row.get('event_first_player')}: {w}W - {l}L"
                      
                 # Process surface
                 stats_p1_surf = process_fixture_surface(df_p1, save_csv=False)
                 if not stats_p1_surf.empty:
                     # Check surface match
                     if surface != "Unknown":
                          # Note: process_files.py uses 'tournament_sourface' column
                          target_key = clean_p1_key
                          row_surf = stats_p1_surf[
                              (stats_p1_surf['results_for_player_key'] == target_key) & 
                              (stats_p1_surf['tournament_sourface'] == surface)
                          ]
                          if not row_surf.empty:
                               w = row_surf.iloc[0]['won_main_player']
                               l = row_surf.iloc[0]['lost_main_player']
                               total = w + l
                               pct = (w / total * 100) if total > 0 else 0
                               p1_surface_recent = f"{pct:.1f}%".replace('.', ',')
                               # Append to recent text
                               p1_recent_text += f" | Sourface: {w}W - {l}L"
        
        # P2
        if p2_key:
             try:
                 clean_p2_key = str(int(float(p2_key)))
             except:
                 clean_p2_key = str(p2_key)

             df_p2 = get_fixtures(date_start=recent_start, date_stop=recent_end, player_key=clean_p2_key, save_json=False, save_csv=False)
             
             if df_p2 is None:
                 api_error = True
             elif not df_p2.empty:
                 # Filter for Singles only
                 if 'event_type_type' in df_p2.columns:
                     df_p2 = df_p2[df_p2['event_type_type'].astype(str).str.contains("Singles", case=False, na=False)]

                 # Capture for detailed view
                 df_p2_all = df_p2.copy()

                 stats_p2 = process_fixture_period(df_p2, save_csv=False)
                 if not stats_p2.empty:
                      # Clean key for matching (handle float string '1234.0' -> '1234')
                      target_key = clean_p2_key
                      row_p2 = stats_p2[stats_p2['results_for_player_key'] == target_key]
                      if not row_p2.empty:
                          w = row_p2.iloc[0]['won_main_player']
                          l = row_p2.iloc[0]['lost_main_player']
                          total = w + l
                          pct = (w / total * 100) if total > 0 else 0
                          p2_recent = f"{pct:.1f}%".replace('.', ',')
                          p2_recent_text = f"{row.get('event_second_player')}: {w}W - {l}L"
                 
                 stats_p2_surf = process_fixture_surface(df_p2, save_csv=False)
                 if not stats_p2_surf.empty:
                     if surface != "Unknown":
                          target_key = clean_p2_key
                          row_surf = stats_p2_surf[
                               (stats_p2_surf['results_for_player_key'] == target_key) & 
                               (stats_p2_surf['tournament_sourface'] == surface)
                          ]
                          if not row_surf.empty:
                               w = row_surf.iloc[0]['won_main_player']
                               l = row_surf.iloc[0]['lost_main_player']
                               total = w + l
                               pct = (w / total * 100) if total > 0 else 0
                               p2_surface_recent = f"{pct:.1f}%".replace('.', ',')
                               # Append to recent text
                               p2_recent_text += f" | Sourface: {w}W - {l}L"

    except Exception as e:
        logging.error(f"Error fetching recent performance: {e}")

    # 4. Build DataFrame
    data = {
        "Date": [row.get("event_date")],
        "Player 1": [row.get("event_first_player")],
        "Player 2": [row.get("event_second_player")],
        "Sourface": [surface],
        "H2H P1": [h2h_p1],
        "H2H % P1": [h2h_p1_pct],
        "H2H P2": [h2h_p2],
        "H2H % P2": [h2h_p2_pct],
        "P1 Rec. Performance": [p1_recent],
        "P1 Sourface R. Perf.": [p1_surface_recent],
        "P1 ATP Points": [p1_points],
        "P2 Rec. Performance": [p2_recent],
        "P2 Sourface R. Perf.": [p2_surface_recent],
        "P2 ATP Points": [p2_points],
    }
    
    df_details = pd.DataFrame(data)
    
    # Display
    st.dataframe(df_details, hide_index=True, use_container_width=True)
    
    # Recent Performance Text
    if p1_recent_text:
        st.caption(p1_recent_text)
    if p2_recent_text:
        st.caption(p2_recent_text)

    # H2H History Table
    if h2h_matches_to_display:
        st.markdown("##### Head-to-Head History (Last 5)")
        st.dataframe(pd.DataFrame(h2h_matches_to_display), hide_index=True, use_container_width=True)
    elif h2h_p1 == 0 and h2h_p2 == 0 and p1_points != "No data":
         # Heuristic: if points are there but H2H is 0, maybe API error?
         # Or just no matches.
         pass
    
    # Recent Matches Section
    selected_player_side = st.session_state.get(f"details_selection_{row.get('event_key', 'unknown')}")
    
    if selected_player_side:
        target_name = row['event_first_player'] if selected_player_side == "P1" else row['event_second_player']
        target_df = df_p1_all if selected_player_side == "P1" else df_p2_all
        
        st.markdown(f"---")
        st.subheader(f"Recent matches of {target_name}")
        
        # Helper to format table
        def get_recent_matches_display(df_matches, target_surface=None):
            if df_matches.empty:
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
    if api_error:
        with error_placeholder.container():
            st.error("API is currently unstable. Please try again in a few moments.")
            if st.button("🔄 Refresh Results"):
                st.rerun()


# ----------------------
# Search Events Section
# ----------------------
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
    search_clicked = st.button("Search", type="primary")

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
            if c5.button("See Details", key=f"btn_details_{row.get('event_key', index)}", type="primary"):
                show_details_dialog(row)
                
            # Add a visual separator
            st.markdown("---")

    else:
        st.info("No matches scheduled for this date.")

