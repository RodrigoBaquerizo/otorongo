import logging
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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    logging.info("Starting the Tennis Standings Fetcher")

    get_standings(event_type="ATP", save_json=True)
    logging.info("Standings data fetched and saved successfully.")

    # get_events(save_json=True)
    # logging.info("Events data fetched and saved successfully.")

    # get_tournaments(save_json=True)
    # logging.info("Tournaments data fetched and saved successfully.")

    # get_fixtures(date_start="2025-11-06", date_stop="2025-11-10", save_json=False)
    # logging.info("Fixtures data fetched and saved successfully.")

    # get_livescore(save_json=True)
    # logging.info("Livescore data fetched and saved successfully.")

    # get_h2h(first_player_key=1107, second_player_key=168, save_json=True)  # or 30 - 5
    # logging.info("H2H data fetched and saved successfully.")

    # get_players(player_key=1107, save_json=True)
    # logging.info("Player data fetched and saved successfully.")

    # get_odds(date_start="2025-12-01", date_stop="2025-12-31", save_json=True)
    # logging.info("Odds data fetched and saved successfully.")

    # get_live_odds(save_json=True)
    # logging.info("Live odds data fetched and saved successfully.")


if __name__ == "__main__":
    main()
