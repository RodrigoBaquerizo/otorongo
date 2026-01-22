import json
import pandas as pd
from datetime import datetime
import requests
import logging
from pathlib import Path
try:
    from scripts.process_files import join_main_files, concat_fixtures, concat_h2h
    from scripts.logger_config import setup_logging
except ImportError:
    from process_files import join_main_files, concat_fixtures, concat_h2h
    from logger_config import setup_logging

import os
from dotenv import load_dotenv

setup_logging()
load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
BASE_URL = f"https://api.api-tennis.com/tennis/?method="


def create_data_folders():
    """Create data directory structure if it doesn't exist."""
    # Define the folder structure
    folders = [Path("data/fixtures"), Path("data/h2h")]

    # Create each folder (including parent directories)
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


create_data_folders()


### HELPER FUNCTIONS ###
def save_to_csv_simple(data: requests.Response, filename: str) -> pd.DataFrame:
    # Create DataFrame from the 'result' list
    df = pd.DataFrame(data["result"])

    # Add download timestamp column
    df["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    df.to_csv(filename, index=False, encoding="utf-8")
    return df


def save_to_json(data: requests.Response, filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


### MAIN API FUNCTIONS ###
def get_events(
    save_json: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_events",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve events for a specific season from the tenis API."""

    authentication = f"&APIkey={api_key}"
    url = base_url + method + authentication

    response = requests.get(url)
    data = response.json()

    if save_json:
        save_to_json(data, "data/get_events.json")

    save_to_csv_simple(data, "data/get_events.csv")


def get_tournaments(
    save_json: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_tournaments",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve tournaments from the tenis API."""

    authentication = f"&APIkey={api_key}"
    url = base_url + method + authentication

    response = requests.get(url)
    data = response.json()

    if save_json:
        save_to_json(data, "data/tournaments.json")

    return save_to_csv_simple(data, "data/tournaments.csv")


## more params needed
def get_fixtures(
    date_start: str,
    date_stop: str,
    player_key: str = "",
    save_json: bool = True,
    save_csv: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_fixtures",
    api_key: str = API_KEY,
    get_pointbypoint: bool = True,
    get_scores: bool = True,
) -> None:
    """Function to retrieve fixtures from the tenis API."""

    search = f"&date_start={date_start}&date_stop={date_stop}"

    file_path = "data/fixtures/fixtures"

    if player_key != "":
        search += f"&player_key={player_key}"
        file_path = "data/fixtures/fixtures" + f"_player_{player_key}"

    authentication = f"&APIkey={api_key}"
    url = base_url + method + search + authentication

    response = requests.get(url)
    if response.status_code == 500:
        logging.error("Server error (500)")
        logging.error(response.text)  # See the full error
        return None

    if response.status_code != 200:
        logging.error(f"HTTP Error {response.status_code}")
        logging.error(response.text)
        return None

    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        logging.error("Response is not valid JSON despite Content-Type header")
        logging.error(response.text[:1000])
        return None

    if save_json:
        save_to_json(data, f"{file_path}.json")

    if not data.get("result"):
        logging.info("No live scores available at the moment.")
        return

    download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Lists to collect data
    events_records = []
    pointbypoint_records = []
    scores_records = []

    for event in data["result"]:
        event_key = event.get("event_key")
        event_date = event.get("event_date")
        
        if not event_key:
            logging.warning(f"Skipping event with missing event_key: {event}")
            continue

        # Save main event data (excluding nested lists)

        event_record = {
            "results_for_player_key": player_key,
            "event_key": event.get("event_key"),
            "event_date": event.get("event_date"),
            "event_time": event.get("event_time"),
            "event_first_player": event.get("event_first_player"),
            "first_player_key": event.get("first_player_key"),
            "event_second_player": event.get("event_second_player"),
            "second_player_key": event.get("second_player_key"),
            "event_final_result": event.get("event_final_result"),
            "event_game_result": event.get("event_game_result"),
            "event_serve": event.get("event_serve"),
            "event_winner": event.get("event_winner"),
            "event_status": event.get("event_status"),
            "event_type_type": event.get("event_type_type"),
            "tournament_name": event.get("tournament_name"),
            "tournament_key": event.get("tournament_key"),
            "tournament_round": event.get("tournament_round"),
            "tournament_season": event.get("tournament_season"),
            "event_live": event.get("event_live"),
            "event_qualification": event.get("event_qualification"),
            "event_first_player_logo": event.get("event_first_player_logo"),
            "event_second_player_logo": event.get("event_second_player_logo"),
            "download_time": download_time,
        }
        events_records.append(event_record)

        if get_pointbypoint:
            # Process pointbypoint with nested points
            for pbp in event.get("pointbypoint", []):
                set_number = pbp.get("set_number")
                number_game = pbp.get("number_game")
                player_served = pbp.get("player_served")
                serve_winner = pbp.get("serve_winner")
                serve_lost = pbp.get("serve_lost")
                game_score = pbp.get("score")

                for point in pbp.get("points", []):
                    point_record = {
                        "event_key": event_key,
                        "event_date": event_date,
                        "set_number": set_number,
                        "number_game": number_game,
                        "player_served": player_served,
                        "serve_winner": serve_winner,
                        "serve_lost": serve_lost,
                        "game_score": game_score,
                        "number_point": point.get("number_point"),
                        "point_score": point.get("score"),
                        "break_point": point.get("break_point"),
                        "set_point": point.get("set_point"),
                        "match_point": point.get("match_point"),
                        "download_time": download_time,
                    }
                    pointbypoint_records.append(point_record)

        if get_scores:
            # Process scores
            for score in event.get("scores", []):
                score_record = {
                    "event_key": event_key,
                    "event_date": event_date,
                    "score_first": score.get("score_first"),
                    "score_second": score.get("score_second"),
                    "score_set": score.get("score_set"),
                    "download_time": download_time,
                }
                scores_records.append(score_record)

    # Save main events data
    # Save main events data
    df_events = pd.DataFrame(events_records)
    if save_csv:
        df_events.to_csv(f"{file_path}.csv", index=False, encoding="utf-8")
    logging.info(f"Fetched {len(df_events)} event records for player {player_key}")

    if pointbypoint_records:
        df_pbp = pd.DataFrame(pointbypoint_records)
        if save_csv:
            df_pbp.to_csv(f"{file_path}_pointbypoint.csv", index=False, encoding="utf-8")
        logging.info(
            f"Fetched {len(df_pbp)} pointbypoint records for player {player_key}"
        )

    if scores_records:
        df_scores = pd.DataFrame(scores_records)
        if save_csv:
            df_scores.to_csv(f"{file_path}_scores.csv", index=False, encoding="utf-8")
        logging.info(f"Fetched {len(df_scores)} scores records for player {player_key}")

    return df_events


## more params needed
def get_livescore(
    save_json: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_livescore",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve live scores from the tenis API"""

    authentication = f"&APIkey={api_key}"
    url = base_url + method + authentication

    response = requests.get(url)
    data = response.json()

    if save_json:
        save_to_json(data, "data/livescore.json")

    if not data.get("result"):
        logging.info("No live scores available at the moment.")
        return

    download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Lists to collect data
    events_records = []
    pointbypoint_records = []
    scores_records = []

    for event in data["result"]:
        event_key = event["event_key"]
        event_date = event["event_date"]

        # Save main event data (excluding nested lists)
        event_record = {
            "event_key": event.get("event_key"),
            "event_date": event.get("event_date"),
            "event_time": event.get("event_time"),
            "event_first_player": event.get("event_first_player"),
            "first_player_key": event.get("first_player_key"),
            "event_second_player": event.get("event_second_player"),
            "second_player_key": event.get("second_player_key"),
            "event_final_result": event.get("event_final_result"),
            "event_game_result": event.get("event_game_result"),
            "event_serve": event.get("event_serve"),
            "event_winner": event.get("event_winner"),
            "event_status": event.get("event_status"),
            "event_type_type": event.get("event_type_type"),
            "tournament_name": event.get("tournament_name"),
            "tournament_key": event.get("tournament_key"),
            "tournament_round": event.get("tournament_round"),
            "tournament_season": event.get("tournament_season"),
            "event_live": event.get("event_live"),
            "event_first_player_logo": event.get("event_first_player_logo"),
            "event_second_player_logo": event.get("event_second_player_logo"),
            "event_qualification": event.get("event_qualification"),
        }
        events_records.append(event_record)

        # Process pointbypoint with nested points
        for pbp in event.get("pointbypoint", []):
            set_number = pbp.get("set_number")
            number_game = pbp.get("number_game")
            player_served = pbp.get("player_served")
            serve_winner = pbp.get("serve_winner")
            serve_lost = pbp.get("serve_lost")
            game_score = pbp.get("score")

            for point in pbp.get("points", []):
                point_record = {
                    "event_key": event_key,
                    "event_date": event_date,
                    "set_number": set_number,
                    "number_game": number_game,
                    "player_served": player_served,
                    "serve_winner": serve_winner,
                    "serve_lost": serve_lost,
                    "game_score": game_score,
                    "number_point": point.get("number_point"),
                    "point_score": point.get("score"),
                    "break_point": point.get("break_point"),
                    "set_point": point.get("set_point"),
                    "match_point": point.get("match_point"),
                    "download_time": download_time,
                }
                pointbypoint_records.append(point_record)

        # Process scores
        for score in event.get("scores", []):
            score_record = {
                "event_key": event_key,
                "event_date": event_date,
                "score_first": score.get("score_first"),
                "score_second": score.get("score_second"),
                "score_set": score.get("score_set"),
                "download_time": download_time,
            }
            scores_records.append(score_record)

    # Save main events data
    df_events = pd.DataFrame(events_records)
    df_events.to_csv("data/livescore.csv", index=False, encoding="utf-8")
    logging.info(f"Saved {len(df_events)} event records")

    if pointbypoint_records:
        df_pbp = pd.DataFrame(pointbypoint_records)
        df_pbp.to_csv("data/livescore_pointbypoint.csv", index=False, encoding="utf-8")
        logging.info(f"Saved {len(df_pbp)} pointbypoint records")

    if scores_records:
        df_scores = pd.DataFrame(scores_records)
        df_scores.to_csv("data/livescore_scores.csv", index=False, encoding="utf-8")
        logging.info(f"Saved {len(df_scores)} scores records")


# wmomre params needed
def get_h2h(
    first_player_key: int,
    second_player_key: int,
    save_json: bool = True,
    save_csv: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_H2H",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve h2h information from the tenis API."""

    search = (
        f"&first_player_key={first_player_key}&second_player_key={second_player_key}"
    )
    authentication = f"&APIkey={api_key}"
    url = base_url + method + search + authentication

    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"HTTP Error {response.status_code} in get_h2h")
        try:
             logging.error(response.text)
        except:
             pass
        return None

    try:
        data = response.json()
    except Exception as e:
        logging.error(f"JSON Error in get_h2h: {e}")
        return None

    if save_json:
        save_to_json(
            data, f"data/h2h/h2h_{first_player_key}_vs_{second_player_key}.json"
        )

    if not data or not data.get("result"):
        logging.info("No H2H available at the moment.")
        return None

    # Create DataFrame from the 'result' list
    df = pd.DataFrame(data["result"]["H2H"])
    # Add download timestamp column
    # Add download timestamp column
    df["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Save to CSV
    if save_csv:
        df.to_csv(
            f"data/h2h/h2h_{first_player_key}_vs_{second_player_key}_h2h.csv",
            index=False,
            encoding="utf-8",
        )

    df = pd.DataFrame(data["result"]["firstPlayerResults"])
    df["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if save_csv:
        df.to_csv(
            f"data/h2h/h2h_{first_player_key}_vs_{second_player_key}_1st_pl.csv",
            index=False,
            encoding="utf-8",
        )

    df = pd.DataFrame(data["result"]["secondPlayerResults"])
    df["download_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if save_csv:
        df.to_csv(
            f"data/h2h/h2h_{first_player_key}_vs_{second_player_key}_2nd_pl.csv",
            index=False,
            encoding="utf-8",
        )

    return data.get("result")


def get_standings(
    event_type: str = "ATP",
    save_json: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_standings",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve standings from the tenis API."""

    search = f"&event_type={event_type}"  # WTA or ATP
    authentication = f"&APIkey={api_key}"
    url = base_url + method + search + authentication

    response = requests.get(url)
    data = response.json()

    if save_json:
        save_to_json(data, f"data/standings_{event_type}.json")

    return save_to_csv_simple(data, f"data/standings_{event_type}.csv")


def get_players(
    player_key: int,
    save_json: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_players",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve player information from the tenis API."""

    search = f"&player_key={player_key}"
    authentication = f"&APIkey={api_key}"
    url = base_url + method + search + authentication

    response = requests.get(url)
    data = response.json()

    if save_json:
        save_to_json(data, f"data/players/players_{player_key}.json")

    # save to csv with nested stats handling
    download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Lists to collect data
    players_records = []
    stats_records = []
    tournaments_records = []

    logging.info(f">> Getting data for player {player_key}...")

    for player in data["result"]:
        player_key = player.get("player_key")

        # Save main player data (without stats)
        player_record = {
            "player_key": player_key,
            "player_name": player.get("player_name"),
            "player_country": player.get("player_country"),
            "player_bday": player.get("player_bday"),
            "player_logo": player.get("player_logo"),
            "download_time": download_time,
        }
        players_records.append(player_record)

        # Process stats
        for stat in player.get("stats", []):
            stat_record = {
                "player_key": player_key,
                "season": stat.get("season"),
                "type": stat.get("type"),
                "rank": stat.get("rank"),
                "titles": stat.get("titles"),
                "matches_won": stat.get("matches_won"),
                "matches_lost": stat.get("matches_lost"),
                "hard_won": stat.get("hard_won"),
                "hard_lost": stat.get("hard_lost"),
                "clay_won": stat.get("clay_won"),
                "clay_lost": stat.get("clay_lost"),
                "grass_won": stat.get("grass_won"),
                "grass_lost": stat.get("grass_lost"),
                "download_time": download_time,
            }
            stats_records.append(stat_record)

        # Process tournaments
        for tournament in player.get("tournaments", []):
            tournament_record = {
                "player_key": player_key,
                "name": tournament.get("name"),
                "season": tournament.get("season"),
                "type": tournament.get("type"),
                "surface": tournament.get("surface"),
                "prize": tournament.get("prize"),
                "download_time": download_time,
            }
            tournaments_records.append(tournament_record)

    # Save players data
    df_players = pd.DataFrame(players_records)
    df_players.to_csv(
        f"data/players/players_{player_key}.csv", index=False, encoding="utf-8"
    )
    logging.info(f"Saved {len(df_players)} player records")

    # Save players stats data
    if stats_records:
        df_stats = pd.DataFrame(stats_records)
        df_stats.to_csv(
            f"data/players/players_{player_key}_stats.csv",
            index=False,
            encoding="utf-8",
        )
        logging.info(f"Saved {len(df_stats)} player stats records")
    else:
        logging.info("No player stats data to save")

    # Save players tournaments data
    if tournaments_records:
        df_tournaments = pd.DataFrame(tournaments_records)
        df_tournaments.to_csv(
            f"data/players/players_{player_key}_tournaments.csv",
            index=False,
            encoding="utf-8",
        )
        logging.info(f"Saved {len(df_tournaments)} player tournaments records")


def get_odds(
    date_start: str,
    date_stop: str,
    save_json: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_odds",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve odds information from the tenis API."""

    search = f"&date_start={date_start}&date_stop={date_stop}"
    authentication = f"&APIkey={api_key}"
    url = base_url + method + search + authentication

    response = requests.get(url)
    data = response.json()

    if save_json:
        save_to_json(data, "data/odds.json")


def get_live_odds(
    save_json: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_live_odds",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve live odds information from the tenis API."""

    search = ""
    authentication = f"&APIkey={api_key}"
    url = base_url + method + search + authentication

    response = requests.get(url)
    data = response.json()

    if save_json:
        save_to_json(data, "data/live_odds.json")


def get_fixtures_for_a_date(
    date: str,
    player_fixture_start="2025-01-01",
    player_fixture_end="2025-12-31",
    tournament_key: int | None = None,
    save_json_all_files: bool = True,
    base_url: str = BASE_URL,
    method: str = "get_fixtures",
    api_key: str = API_KEY,
) -> None:
    """Function to retrieve fixtures from the tenis API for a specific date."""

    search = f"&date_start={date}&date_stop={date}"
    if tournament_key is not None:
        search += f"&tournament_key={tournament_key}"

    authentication = f"&APIkey={api_key}"
    url = base_url + method + search + authentication

    response = requests.get(url)

    data = response.json()

    if save_json_all_files:
        save_to_json(data, f"data/fixtures_for_date_{date}.json")

    if not data.get("result"):
        logging.info(f"No fixtures available for the date {date}.")
        return

    download_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_mssg = f"""

    Getting data for date: {date}
    Tournament key filter applied: {"all" if tournament_key is None else tournament_key}
    Saving all intermediate JSON files: {save_json_all_files}
    """
    logging.info(log_mssg)

    # Lists to collect data
    events_records = []
    players_list = []

    logging.info(f"Getting event data for date {date}...")

    for event in data["result"]:
        # event_key = event["event_key"]
        # event_date = event["event_date"]

        # Save main event data (excluding nested lists)
        event_record = {
            "event_key": event.get("event_key"),
            "event_date": event.get("event_date"),
            "event_time": event.get("event_time"),
            "event_first_player": event.get("event_first_player"),
            "first_player_key": event.get("first_player_key"),
            "event_second_player": event.get("event_second_player"),
            "second_player_key": event.get("second_player_key"),
            "event_final_result": event.get("event_final_result"),
            "event_game_result": event.get("event_game_result"),
            "event_serve": event.get("event_serve"),
            "event_winner": event.get("event_winner"),
            "event_status": event.get("event_status"),
            "event_type_type": event.get("event_type_type"),
            "tournament_name": event.get("tournament_name"),
            "tournament_key": event.get("tournament_key"),
            "tournament_round": event.get("tournament_round"),
            "tournament_season": event.get("tournament_season"),
            "event_live": event.get("event_live"),
            "event_qualification": event.get("event_qualification"),
            "event_first_player_logo": event.get("event_first_player_logo"),
            "event_second_player_logo": event.get("event_second_player_logo"),
            "download_time": download_time,
        }
        events_records.append(event_record)

        players_list.append(event.get("first_player_key"))
        players_list.append(event.get("second_player_key"))

        get_h2h(
            first_player_key=event.get("first_player_key"),
            second_player_key=event.get("second_player_key"),
            save_json=save_json_all_files,
        )

    # Save main events data
    df_events = pd.DataFrame(events_records)
    df_events.to_csv(
        f"data/fixtures_for_date_{date}.csv", index=False, encoding="utf-8"
    )
    logging.info(f"Saved {len(df_events)} event records")
    logging.info("------")

    # ## getting player data
    # logging.info(f"Getting player data...")
    # for player in set(players_list):
    #     get_players(player_key=player, save_json=save_json_all_files)
    # logging.info(f"Fetched and saved data for {len(set(players_list))} players")
    # logging.info("------")

    ## getting tournaments data
    logging.info(f"Getting tournaments data...")
    get_tournaments(save_json=True)
    logging.info("Fetched and saved tournaments data")
    logging.info("------")

    ## getting fixtures data for each player
    logging.info("Getting fixtures data for all players...")
    for player in set(players_list):
        logging.info(f">> Getting fixture dta for player {player}...")
        get_fixtures(
            date_start=player_fixture_start,
            date_stop=player_fixture_end,
            player_key=player,
            save_json=save_json_all_files,
            get_pointbypoint=False,
            get_scores=False,
        )
    logging.info(
        f"Fetched and saved fixtures data for {len(set(players_list))} players"
    )
    logging.info("------")

    ## geting rankings
    logging.info("Getting rankings data...")
    get_standings(event_type="ATP", save_json=save_json_all_files)
    get_standings(event_type="WTA", save_json=save_json_all_files)
    logging.info("------")

    logging.info("All data fetching for fixtures on the date completed.")
    logging.info("------")
    logging.info("------")

    logging.info("Starting file processing...")

    logging.info("Joining and processing player's fixtures...")
    concat_fixtures()
    logging.info("------")

    logging.info("Joining and processing H2H files...")
    concat_h2h()
    logging.info("------")

    logging.info("Starting final process...")
    join_main_files(for_date=date, fixtures_file=f"data/fixtures_for_date_{date}.csv")
    logging.info("Process completed.")
    logging.info("------")


if __name__ == "__main__":
    get_fixtures_for_a_date(
        date="2026-01-07", save_json_all_files=True, tournament_key=8455
    )
