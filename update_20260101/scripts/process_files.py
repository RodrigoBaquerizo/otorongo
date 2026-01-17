import pandas as pd
import os
import numpy as np
from typing import List, Dict
from my_columns import COLUMNS
import logging
from logger_config import setup_logging

setup_logging()


H2H_PROCESSED_FILE = "final_h2h.csv"
PLAYERS_FIXTURE_ALL = "concat_players_fixture.csv"
PLAYERS_FIXTURE_PERIOD = "final_players_fixture.csv"
PLAYERS_FIXTURE_SURFACE = "final_by_surface_players_fixture.csv"


def concat_h2h(folder: str = "h2h", output_file: str = H2H_PROCESSED_FILE):
    file_list = [
        f"data/{folder}/{x}"
        for x in os.listdir(f"data/{folder}/")
        if x.endswith("h2h.csv")
    ]

    dfs = []
    dfs_check = []

    for file in file_list:
        df = pd.read_csv(file)
        if df.empty:
            continue

        dfs_check.append(df)

        df.dropna(
            inplace=True, subset=["event_winner"]
        )  ## needed it not could give wrong win/lose results

        df["event_key"] = df["event_key"].astype("int")
        df["first_player_key"] = df["first_player_key"].astype("int")
        df["second_player_key"] = df["second_player_key"].astype("int")

        players = pd.Series(
            df["first_player_key"].tolist() + df["second_player_key"].tolist()
        ).unique()
        df2 = df.assign(
            winner_key=np.where(
                df["event_winner"] == "First Player",
                df["first_player_key"],
                df["second_player_key"],
            )
        )
        df2 = (
            df2.groupby("winner_key")["event_key"]
            .count()
            .reindex(players, fill_value=0)
            .reset_index()
        )
        df2.columns = ["winner_key", "event_key"]

        result = pd.DataFrame(
            {
                "first_player_key": [
                    df2.iloc[0]["winner_key"],
                    df2.iloc[1]["winner_key"],
                ],
                "total_first_player_key": [
                    df2.iloc[0]["event_key"],
                    df2.iloc[1]["event_key"],
                ],
                "second_player_key": [
                    df2.iloc[1]["winner_key"],
                    df2.iloc[0]["winner_key"],
                ],
                "total_second_player_key": [
                    df2.iloc[1]["event_key"],
                    df2.iloc[0]["event_key"],
                ],
            }
        )
        dfs.append(result)

    df_concat = pd.concat(dfs).drop_duplicates().reset_index(drop=True)
    df_concat.to_csv(f"data/{output_file}", index=False, encoding="utf-8")

    df_concat2 = pd.concat(dfs_check).drop_duplicates().reset_index(drop=True)
    df_concat2.to_csv("data/concat_h2h_check.csv", index=False, encoding="utf-8")


def process_fixture_period(df, output_file=PLAYERS_FIXTURE_PERIOD):
    df["event_date"] = pd.to_datetime(df["event_date"])
    df2 = (
        df.assign(
            winner_key=lambda x: np.where(
                x["event_winner"] == "First Player",
                x["first_player_key"],
                x["second_player_key"],
            ),
            won_main_player=lambda x: np.where(
                x["results_for_player_key"] == x["winner_key"], 1, 0
            ),
            lost_main_player=lambda x: np.where(
                x["results_for_player_key"] != x["winner_key"], 1, 0
            ),
        )
        .groupby(["results_for_player_key"])
        .agg(
            {"won_main_player": "sum", "lost_main_player": "sum", "event_key": "count"}
        )
        .reset_index()
    )  # count separado para validar

    df2.to_csv(f"data/{output_file}", index=False, encoding="utf-8")


def process_fixture_surface(df, output_file=PLAYERS_FIXTURE_SURFACE):
    df_trn = pd.read_csv(
        "data/tournaments.csv", usecols=["tournament_key", "tournament_sourface"]
    )
    joined = df.merge(
        df_trn,
        left_on="tournament_key",
        right_on="tournament_key",
        how="left",
        suffixes=("", "_trn"),
    )
    df2 = (
        joined.assign(
            winner_key=lambda x: np.where(
                x["event_winner"] == "First Player",
                x["first_player_key"],
                x["second_player_key"],
            ),
            won_main_player=lambda x: np.where(
                x["results_for_player_key"] == x["winner_key"], 1, 0
            ),
            lost_main_player=lambda x: np.where(
                x["results_for_player_key"] != x["winner_key"], 1, 0
            ),
        )
        .groupby(["results_for_player_key", "tournament_sourface"])
        .agg(
            {"won_main_player": "sum", "lost_main_player": "sum", "event_key": "count"}
        )
        .reset_index()
    )

    df2.to_csv(f"data/{output_file}", index=False, encoding="utf-8")


def concat_fixtures(folder: str = "fixtures", output_file: str = PLAYERS_FIXTURE_ALL):
    file_list = [
        f"data/{folder}/{x}"
        for x in os.listdir(f"data/{folder}/")
        if x.endswith(".csv")
    ]
    dfs = [pd.read_csv(file) for file in file_list]
    df_concat = (
        pd.concat(dfs)
        .drop(["download_time"], axis=1)
        .drop_duplicates()
        .dropna(
            subset=["event_winner"]
        )  ## needed it not could give wrong win/lose results
        .reset_index(drop=True)
    )
    df_concat.to_csv(f"data/{output_file}", index=False, encoding="utf-8")
    logging.info(
        f"Concatenated {len(file_list)} files into data/{output_file} with shape {df_concat.shape}"
    )

    process_fixture_period(df_concat)

    process_fixture_surface(df_concat)


def apply_column_config(df: pd.DataFrame, columns_config: List[Dict]) -> pd.DataFrame:
    """
    Apply column configuration to a DataFrame: select columns and rename as specified.

    Args:
        df: Source DataFrame
        columns_config: List of column configuration dictionaries

    Returns:
        DataFrame with selected and renamed columns
    """
    # Filter columns that are in use
    cols_in_use = [col["name"] for col in columns_config if col["in_use"]]

    # Select only the columns that are in use
    df_filtered = df[cols_in_use].copy()

    # Create rename mapping for columns that need renaming
    rename_map = {
        col["name"]: col["new_name"]
        for col in columns_config
        if col["in_use"] and col["rename"]
    }

    # Apply renaming
    if rename_map:
        df_filtered = df_filtered.rename(columns=rename_map)

    return df_filtered


def join_main_files(
    for_date: str,
    fixtures_file: str,
    h2h_file: str = f"data/{H2H_PROCESSED_FILE}",
    torunament_file: str = "data/tournaments.csv",
    players_results_file: str = f"data/{PLAYERS_FIXTURE_PERIOD}",
    player_surface_file: str = f"data/{PLAYERS_FIXTURE_SURFACE}",
):
    # read h2h concatenated file and matches of the day
    df_h2h = pd.read_csv(h2h_file)
    df_sch = pd.read_csv(fixtures_file)
    df_trn = pd.read_csv(
        torunament_file, usecols=["tournament_key", "tournament_sourface"]
    )
    df_players_results = pd.read_csv(players_results_file)
    df_srfc = pd.read_csv(player_surface_file)

    standing_dfs = []
    for file in os.listdir("data/"):
        if file.startswith("standings_") and file.endswith(".csv"):
            df_st_ = pd.read_csv(f"data/{file}", usecols=["player_key", "points"])
            standing_dfs.append(df_st_)
    df_st = pd.concat(standing_dfs).drop_duplicates().reset_index(drop=True)

    joined = df_sch.merge(
        df_h2h,
        on=["first_player_key", "second_player_key"],
        how="left",
        suffixes=("", "_h2h"),
    ).merge(
        df_trn,
        left_on="tournament_key",
        right_on="tournament_key",
        how="left",
        suffixes=("", "_trn"),
    )

    # Fill NaNs in columns from df_h2h with 0
    h2h_cols = [
        col
        for col in df_h2h.columns
        if col not in ["first_player_key", "second_player_key", "tournament_sourface"]
    ]
    joined[h2h_cols] = joined[h2h_cols].fillna(0)

    # joined.to_csv("data/joined_schedule_h2h.csv", index=False, encoding="utf-8")

    joined2 = (
        joined.merge(
            df_players_results,
            left_on="first_player_key",
            right_on="results_for_player_key",
            how="left",
            suffixes=("", "_first"),
        )
        .merge(
            df_players_results,
            left_on="second_player_key",
            right_on="results_for_player_key",
            how="left",
            suffixes=("", "_second"),
        )
        .merge(
            df_srfc,
            left_on=["first_player_key", "tournament_sourface"],
            right_on=["results_for_player_key", "tournament_sourface"],
            how="left",
            suffixes=("", "_first_surface"),
        )
        .merge(
            df_srfc,
            left_on=["second_player_key", "tournament_sourface"],
            right_on=["results_for_player_key", "tournament_sourface"],
            how="left",
            suffixes=("", "_second_surface"),
        )
        .merge(
            df_st,
            left_on="first_player_key",
            right_on="player_key",
            how="left",
            suffixes=("", "_first_standing"),
        )
        .merge(
            df_st,
            left_on="second_player_key",
            right_on="player_key",
            how="left",
            suffixes=("", "_second_standing"),
        )
    )

    joined2.to_csv("data/full_all_columns_final.csv", index=False, encoding="utf-8")

    final_df = apply_column_config(joined2, COLUMNS)

    final_df.to_csv(
        f"data/FINAL_for_date_{for_date}.csv", index=False, encoding="utf-8"
    )


if __name__ == "__main__":
    date = "2026-01-02"
    logging.info("Starting file processing...")
    concat_fixtures()

    logging.info("Starting concatenating H2H files...")
    concat_h2h()

    logging.info("Starting joining files...")
    join_main_files(for_date=date, fixtures_file=f"data/fixtures_for_date_{date}.csv")
    logging.info("Done")
