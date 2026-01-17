import pandas as pd
import os

MAIN_FILE = "../data/player_ids_fixtures.csv"
EXTRACT_FILE = "../data/fixtures__.csv"


df_first = pd.read_csv(
    EXTRACT_FILE, usecols=["event_first_player", "first_player_key"]
).rename(
    columns={"event_first_player": "player_name", "first_player_key": "player_key"}
)
df_second = pd.read_csv(
    EXTRACT_FILE, usecols=["event_second_player", "second_player_key"]
).rename(
    columns={"event_second_player": "player_name", "second_player_key": "player_key"}
)

print(df_first.shape)
print(df_second.shape)

concat_dfs = [df_first, df_second]

if os.path.exists(MAIN_FILE):
    df_main = pd.read_csv(MAIN_FILE)
    concat_dfs.append(df_main)
    print(df_main.shape)

df_players = pd.concat(concat_dfs).drop_duplicates().reset_index(drop=True)

print(df_players.shape)
print(df_players.head(10))


df_players.to_csv(MAIN_FILE, index=False, encoding="utf-8")
