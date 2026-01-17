# Info

#### API docs

https://api-tennis.com/documentation#fixtures

## Logic

The function **get_fixtures_for_a_date** takes the following arguments:

- date: text, e.g., '2025-12-20'
- player_fixture_start: text, e.g., '2025-01-01' — date from which to start retrieving a player's matches
- player_fixture_end: text, e.g., '2025-12-31' — date up to which to retrieve a player's matches
- tournament_key: integer or None, e.g., 5455. If None, all tournaments are downloaded. Default: None
- save_json_all_files: True or False. If True, saves all data extracted from the API as JSON files. Default: True
- base_url: text. Base URL for the API. Default: https://api.api-tennis.com/tennis/?method=
- method: text. API method used to get the data. Default: 'get_fixtures'
- api_key: text. API key.

### Process

1. Downloads fixtures (matches) for a given date and, optionally, a specific tournament_key. Final file: 'fixtures_for_date_yyyy-mm-dd.csv'.
2. After fetching the matches, downloads the head-to-head (h2h) data for each match into the '\_h2h/' folder.
3. Saves a list of all players involved into the 'fixtures/' folder.
4. Retrieves data for all tournaments. File: 'tournaments.csv'.
5. Downloads fixtures (matches) for each player over the specified period.
6. Downloads rankings/standings data for both 'ATP' and 'WTA'. Files: 'standings_ATP.csv' and 'standings_WTA.csv'.
7. Once all data has been downloaded, begins processing the files.
8. Using the **concat_fixtures** function, concatenates all players' fixture files. Then re-processes the combined file to produce: (a) each player's fixture for the entire period, and (b) each player's fixture for the entire period, grouped by surface. Final files: 'concat_players_fixture.csv', 'final_players_fixture.csv', and 'final_by_surface_players_fixture.csv'.
9. Using the **concat_h2h** function, concatenates all h2h files and summarizes each h2h to compute totals. Final files: 'concat_h2h_check.csv' and 'final_h2h.csv'.
10. Finally, joins previously processed files using the **join_main_files** function to create 'full_all_columns_final.csv'. Then uses 'my_columns.py' to select and rename columns for the final file 'FINAL_for_date_yyyy-mm-dd.csv'.
