import streamlit as st
import logging
import pandas as pd
from datetime import datetime
from scripts.tenis_api import (
    get_standings,
    get_events,
    get_tournaments,
    get_fixtures,
    get_livescore,
    get_h2h,
    get_players,
    get_odds,
    get_live_odds,
)
from scripts.process_files import process_fixture_period, process_fixture_surface

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

st.set_page_config(page_title="Tennis API Data Fetcher", layout="wide")

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
def load_tournaments_data():
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
    
    st.write(f"Details for **{row['event_first_player']}** vs **{row['event_second_player']}**")
    
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
            df_tournaments = load_tournaments_data()
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
    
    # Copy functionality
    if st.button("📋 Copy for Excel"):
        try:
            # Copy to clipboard (tab-separated for Excel)
            df_details.to_csv(sep='\t', index=False, header=False, path_or_buf=None)
            df_details.to_clipboard(sep='\t', index=False, header=False)
            st.toast("✅ Copied to clipboard!", icon="📋")
        except Exception as e:
            st.error(f"Could not copy to clipboard: {e}")
            # Fallback: show code block if clipboard access fails
            csv_string = df_details.to_csv(sep='\t', index=False, header=False)
            st.code(csv_string, language="text")

    # Display API Error if flagged
    if api_error:
        with error_placeholder.container():
            st.error("API is currently unstable. Please try again in a few moments.")
            if st.button("🔄 Refresh Results"):
                st.rerun()


st.subheader("Individual Functions")
# Create tabs for each function
tab_search, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Search Events",
        "Standings",
        "Events",
        "Tournaments",
        "Fixtures",
        "Livescore",
        "H2H",
        "Players",
        "Odds",
        "Live Odds",
    ]
)

# Tab: Search Events
with tab_search:
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

# Tab 1: Standings
with tab1:
    st.markdown("### 📊 Get Standings")
    col1, col2 = st.columns([3, 1])
    with col1:
        event_type = st.selectbox(
            "Event Type", ["ATP", "WTA", "ITF"], key="standings_event"
        )
    with col2:
        save_json_standings = st.checkbox("Save JSON", value=True, key="standings_save")

    if st.button("Run Get Standings", key="btn_standings"):
        with st.spinner("Fetching standings..."):
            try:
                get_standings(event_type=event_type, save_json=save_json_standings)
                st.success("✅ Standings data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 2: Events
with tab2:
    st.markdown("### 🎪 Get Events")
    save_json_events = st.checkbox("Save JSON", value=True, key="events_save")

    if st.button("Run Get Events", key="btn_events"):
        with st.spinner("Fetching events..."):
            try:
                get_events(save_json=save_json_events)
                st.success("✅ Events data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 3: Tournaments
with tab3:
    st.markdown("### 🏆 Get Tournaments")
    save_json_tournaments = st.checkbox("Save JSON", value=True, key="tournaments_save")

    if st.button("Run Get Tournaments", key="btn_tournaments"):
        with st.spinner("Fetching tournaments..."):
            try:
                get_tournaments(save_json=save_json_tournaments)
                st.success("✅ Tournaments data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 4: Fixtures
with tab4:
    st.markdown("### 📅 Get Fixtures")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        date_start_fixtures = st.date_input(
            "Start Date", value=datetime(2025, 12, 1), key="fixtures_start"
        )
    with col2:
        date_stop_fixtures = st.date_input(
            "Stop Date", value=datetime(2025, 12, 31), key="fixtures_stop"
        )
    with col3:
        save_json_fixtures = st.checkbox("Save JSON", value=True, key="fixtures_save")

    if st.button("Run Get Fixtures", key="btn_fixtures"):
        with st.spinner("Fetching fixtures..."):
            try:
                get_fixtures(
                    date_start=date_start_fixtures.strftime("%Y-%m-%d"),
                    date_stop=date_stop_fixtures.strftime("%Y-%m-%d"),
                    save_json=save_json_fixtures,
                )
                st.success("✅ Fixtures data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 5: Livescore
with tab5:
    st.markdown("### 🔴 Get Livescore")
    save_json_livescore = st.checkbox("Save JSON", value=True, key="livescore_save")

    if st.button("Run Get Livescore", key="btn_livescore"):
        with st.spinner("Fetching livescore..."):
            try:
                get_livescore(save_json=save_json_livescore)
                st.success("✅ Livescore data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 6: H2H
with tab6:
    st.markdown("### 🆚 Get Head-to-Head")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        first_player_key = st.number_input(
            "First Player Key", value=1107, step=1, key="h2h_first"
        )
    with col2:
        second_player_key = st.number_input(
            "Second Player Key", value=168, step=1, key="h2h_second"
        )
    with col3:
        save_json_h2h = st.checkbox("Save JSON", value=True, key="h2h_save")

    if st.button("Run Get H2H", key="btn_h2h"):
        with st.spinner("Fetching H2H data..."):
            try:
                get_h2h(
                    first_player_key=int(first_player_key),
                    second_player_key=int(second_player_key),
                    save_json=save_json_h2h,
                )
                st.success("✅ H2H data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 7: Players
with tab7:
    st.markdown("### 👤 Get Players")
    col1, col2 = st.columns([3, 1])
    # with col1:
    #     player_key = st.number_input(
    #         "Player Key", value=1905, step=1, key="players_key"
    #     )
    with col1:
        player_keys = st.text_input(
            "Player Key(s) (comma-separated)", value="1905", key="players_key"
        )
    with col2:
        save_json_players = st.checkbox("Save JSON", value=True, key="players_save")

    # if st.button("Run Get Players", key="btn_players"):
    #     with st.spinner("Fetching player data..."):
    #         try:
    #             get_players(player_key=int(player_key), save_json=save_json_players)
    #             st.success("✅ Player data fetched successfully!")
    #         except Exception as e:
    #             st.error(f"❌ Error: {str(e)}")
    if st.button("Run Get Players", key="btn_players"):
        with st.spinner("Fetching player data..."):
            try:
                # Split by comma, strip whitespace, and convert to integers
                keys = [int(k.strip()) for k in player_keys.split(",") if k.strip()]

                for idx, key in enumerate(keys, 1):
                    get_players(player_key=key, save_json=save_json_players)
                    st.success(
                        f"✅ Player data fetched for key {key} ({idx}/{len(keys)})"
                    )

            except ValueError:
                st.error("❌ Error: Please enter valid numeric player keys")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 8: Odds
with tab8:
    st.markdown("### 💰 Get Odds")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        date_start_odds = st.date_input(
            "Start Date", value=datetime(2025, 12, 1), key="odds_start"
        )
    with col2:
        date_stop_odds = st.date_input(
            "Stop Date", value=datetime(2025, 12, 31), key="odds_stop"
        )
    with col3:
        save_json_odds = st.checkbox("Save JSON", value=True, key="odds_save")

    if st.button("Run Get Odds", key="btn_odds"):
        with st.spinner("Fetching odds..."):
            try:
                get_odds(
                    date_start=date_start_odds.strftime("%Y-%m-%d"),
                    date_stop=date_stop_odds.strftime("%Y-%m-%d"),
                    save_json=save_json_odds,
                )
                st.success("✅ Odds data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 9: Live Odds
with tab9:
    st.markdown("### 💸 Get Live Odds")
    save_json_live_odds = st.checkbox("Save JSON", value=True, key="live_odds_save")

    if st.button("Run Get Live Odds", key="btn_live_odds"):
        with st.spinner("Fetching live odds..."):
            try:
                get_live_odds(save_json=save_json_live_odds)
                st.success("✅ Live odds data fetched successfully!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
