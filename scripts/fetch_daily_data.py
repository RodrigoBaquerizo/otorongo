import os
import sys
from pathlib import Path

# Add project root to path so we can import from scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tenis_api import get_tournaments, get_standings
from scripts.logger_config import setup_logging
import logging

def fetch_daily_data():
    setup_logging()
    logging.info("🚀 Starting Daily Data Fetch...")

    # 1. Update Tournaments (Surfaces, etc.)
    logging.info("Fetching Tournaments...")
    try:
        get_tournaments(save_json=False)
        logging.info("✅ Tournaments updated.")
    except Exception as e:
        logging.error(f"❌ Error fetching tournaments: {e}")

    # 2. Update Rankings
    logging.info("Fetching ATP Standings...")
    try:
        get_standings(event_type="ATP", save_json=False)
        logging.info("✅ ATP Standings updated.")
    except Exception as e:
        logging.error(f"❌ Error fetching ATP standings: {e}")

    logging.info("Fetching WTA Standings...")
    try:
        get_standings(event_type="WTA", save_json=False)
        logging.info("✅ WTA Standings updated.")
    except Exception as e:
        logging.error(f"❌ Error fetching WTA standings: {e}")

    logging.info("🏁 Daily Data Fetch Complete.")

if __name__ == "__main__":
    fetch_daily_data()
