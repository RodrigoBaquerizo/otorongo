import pandas as pd
import requests
import os
import time
import logging
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timedelta
import unidecode

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_name(name):
    if not isinstance(name, str): return ""
    return " ".join(unidecode.unidecode(name).lower().replace(".", " ").strip().split())

def match_names(excel_name, api_name):
    e_clean = clean_name(excel_name)
    a_clean = clean_name(api_name)
    if e_clean == a_clean: return True
    e_parts = e_clean.split()
    a_parts = a_clean.split()
    if len(e_parts) >= 2 and len(a_parts) >= 2:
        if e_parts[0][0] == a_parts[0][0] and e_parts[-1] == a_parts[-1]: return True
    return False

def main():
    load_dotenv(override=True)
    api_key = os.getenv("API_KEY")
    excel_path = Path("data/missing_player_key.xlsx")
    df = pd.read_excel(excel_path)
    if "player_key" not in df.columns: df["player_key"] = None
    
    targets = []
    for i, row in df.iterrows():
        if pd.isna(row.get("player_key")):
            targets.append((i, row.get("player_name", "")))

    logging.info(f"Buscando {len(targets)} jugadores en el Barrido Final 2024...")

    # Chunks de 7 días para 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 1, 1)
    current_start = start_date
    delta = timedelta(days=7)
    
    total_found = 0
    while current_start < end_date and targets:
        s_str = current_start.strftime("%Y-%m-%d")
        e_str = (current_start + delta).strftime("%Y-%m-%d")
        logging.info(f"📅 2024 Bloque: {s_str} a {e_str}")
        
        url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&date_start={s_str}&date_stop={e_str}&APIkey={api_key}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                data = r.json()
                fixtures = data.get("result", [])
                for match in fixtures:
                    for prefix in ["first", "second"]:
                        p_name = match.get(f"event_{prefix}_player")
                        p_key = match.get(f"{prefix}_player_key")
                        if p_name and p_key:
                            matched_idx = -1
                            for i, (idx, t_name) in enumerate(targets):
                                if match_names(t_name, p_name):
                                    df.at[idx, "player_key"] = str(p_key)
                                    df.at[idx, "player_country"] = match.get("tournament_name")
                                    total_found += 1
                                    logging.info(f"  ✨ [2024 MATCH] {t_name} <-> {p_name}")
                                    matched_idx = i
                                    break
                            if matched_idx >= 0: targets.pop(matched_idx)
            if total_found > 0: df.to_excel(excel_path, index=False)
        except: pass
        current_start += delta + timedelta(days=1)
        time.sleep(0.4)

    df.to_excel(excel_path, index=False)
    logging.info(f"Finalizado. Total 2024: {total_found}")

if __name__ == "__main__":
    main()
