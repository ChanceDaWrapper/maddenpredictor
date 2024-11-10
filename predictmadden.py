import pandas as pd
import math
import numpy as np
import time
from collections import defaultdict

start_time = time.time()

# Current Season Index (number of years in - 1)
current_season = 1
# Load the CSV data
# Load the data
df = pd.read_csv('MC_games.csv')

# Define divisions and conferences
divisions = {
    'Bills': 'AFC East', 'Patriots': 'AFC East', 'Dolphins': 'AFC East', 'Jets': 'AFC East',
    'Browns': 'AFC North', 'Steelers': 'AFC North', 'Ravens': 'AFC North', 'Bengals': 'AFC North',
    'Broncos': 'AFC West', 'Chiefs': 'AFC West', 'Raiders': 'AFC West', 'Chargers': 'AFC West',
    'Texans': 'AFC South', 'Colts': 'AFC South', 'Jaguars': 'AFC South', 'Titans': 'AFC South',
    'Giants': 'NFC East', 'Eagles': 'NFC East', 'Cowboys': 'NFC East', 'Commanders': 'NFC East',
    'Rams': 'NFC West', 'Cardinals': 'NFC West', '49ers': 'NFC West', 'Seahawks': 'NFC West',
    'Lions': 'NFC North', 'Vikings': 'NFC North', 'Packers': 'NFC North', 'Bears': 'NFC North',
    'Falcons': 'NFC South', 'Panthers': 'NFC South', 'Saints': 'NFC South', 'Buccaneers': 'NFC South'
}

conferences = {
    'Bills': 'AFC', 'Patriots': 'AFC', 'Dolphins': 'AFC', 'Jets': 'AFC',
    'Browns': 'AFC', 'Steelers': 'AFC', 'Ravens': 'AFC', 'Bengals': 'AFC',
    'Broncos': 'AFC', 'Chiefs': 'AFC', 'Raiders': 'AFC', 'Chargers': 'AFC',
    'Texans': 'AFC', 'Colts': 'AFC', 'Jaguars': 'AFC', 'Titans': 'AFC',
    'Giants': 'NFC', 'Eagles': 'NFC', 'Cowboys': 'NFC', 'Commanders': 'NFC',
    'Rams': 'NFC', 'Cardinals': 'NFC', '49ers': 'NFC', 'Seahawks': 'NFC',
    'Lions': 'NFC', 'Vikings': 'NFC', 'Packers': 'NFC', 'Bears': 'NFC',
    'Falcons': 'NFC', 'Panthers': 'NFC', 'Saints': 'NFC', 'Buccaneers': 'NFC'
}

# Add division and conference columns to the DataFrame
df['home_team_division'] = df['homeTeam'].map(divisions)
df['away_team_division'] = df['awayTeam'].map(divisions)
df['home_team_conference'] = df['homeTeam'].map(conferences)
df['away_team_conference'] = df['awayTeam'].map(conferences)

# Clean the data
future_games = df[(df['stageIndex'] == 1) & (df['status'] == 1)]
test = df[(df['stageIndex'] == 1) & (df['status'] == 1)]
df = df[(df['stageIndex'] == 1) & (df['status'] > 1)]
df_current_season = df[(df['seasonIndex'] == max(df['seasonIndex']))]

# Initialize a dictionary to store team records
team_records = {}
head_to_head_results = {}

# Process the data to calculate win/loss records
for index, row in df.iterrows():
    home_team = row['homeTeam']
    away_team = row['awayTeam']
    home_score = row['homeScore']
    away_score = row['awayScore']
    season_index = row['seasonIndex']
    current_season = max(df['seasonIndex'])  # Assuming you want to track the most recent season as the current season

    if home_team not in team_records:
        team_records[home_team] = {
            'current_season': {
                'wins': 0, 'losses': 0, 'games': 0,
                'division_wins': 0, 'division_losses': 0,
                'conference_wins': 0, 'conference_losses': 0,
                'common_games': {'wins': 0, 'losses': 0, 'games': 0}
            },
            'previous_seasons': {'wins': 0, 'losses': 0, 'games': 0},
        }
    if away_team not in team_records:
        team_records[away_team] = {
            'current_season': {
                'wins': 0, 'losses': 0, 'games': 0,
                'division_wins': 0, 'division_losses': 0,
                'conference_wins': 0, 'conference_losses': 0,
                'common_games': {'wins': 0, 'losses': 0, 'games': 0}
            },
            'previous_seasons': {'wins': 0, 'losses': 0, 'games': 0},
        }

    # Initialize head-to-head records if not already set
    if (home_team, away_team) not in head_to_head_results:
        head_to_head_results[(home_team, away_team)] = {'wins': 0, 'losses': 0}
        head_to_head_results[(away_team, home_team)] = {'wins': 0, 'losses': 0}

    # Update the records based on the scores
    if home_score > away_score:
        if season_index == current_season:
            team_records[home_team]['current_season']['wins'] += 1
            team_records[away_team]['current_season']['losses'] += 1
            head_to_head_results[(home_team, away_team)]['wins'] += 1
            head_to_head_results[(away_team, home_team)]['losses'] += 1
            if row['home_team_division'] == row['away_team_division']:
                team_records[home_team]['current_season']['division_wins'] += 1
                team_records[away_team]['current_season']['division_losses'] += 1
            if row['home_team_conference'] == row['away_team_conference']:
                team_records[home_team]['current_season']['conference_wins'] += 1
                team_records[away_team]['current_season']['conference_losses'] += 1
        else:
            team_records[home_team]['previous_seasons']['wins'] += 1
            team_records[away_team]['previous_seasons']['losses'] += 1
    elif home_score < away_score:
        if season_index == current_season:
            team_records[away_team]['current_season']['wins'] += 1
            team_records[home_team]['current_season']['losses'] += 1
            head_to_head_results[(away_team, home_team)]['wins'] += 1
            head_to_head_results[(home_team, away_team)]['losses'] += 1
            if row['home_team_division'] == row['away_team_division']:
                team_records[away_team]['current_season']['division_wins'] += 1
                team_records[home_team]['current_season']['division_losses'] += 1
            if row['home_team_conference'] == row['away_team_conference']:
                team_records[away_team]['current_season']['conference_wins'] += 1
                team_records[home_team]['current_season']['conference_losses'] += 1
        else:
            team_records[away_team]['previous_seasons']['wins'] += 1
            team_records[home_team]['previous_seasons']['losses'] += 1

    # Increment the game count for both teams
    if season_index == current_season:
        team_records[home_team]['current_season']['games'] += 1
        team_records[away_team]['current_season']['games'] += 1
    else:
        team_records[home_team]['previous_seasons']['games'] += 1
        team_records[away_team]['previous_seasons']['games'] += 1

# Calculate win percentages
for team, record in team_records.items():
    current_season_record = record['current_season']
    previous_seasons_record = record['previous_seasons']
    
    # Calculate current season win percentage
    current_win_percentage = current_season_record['wins'] / current_season_record['games'] if current_season_record['games'] > 0 else 0
    
    # Calculate past seasons win percentage
    past_win_percentage = previous_seasons_record['wins'] / previous_seasons_record['games'] if previous_seasons_record['games'] > 0 else 0
    
    # Store the win percentages in the record
    record['current_win_percentage'] = current_win_percentage
    record['past_win_percentage'] = past_win_percentage
    
    # Calculate the overall win percentage as an average of the current and past win percentages
    record['win_percentage'] = (current_win_percentage + past_win_percentage) / 2




# Create a DataFrame for team records
team_df = pd.DataFrame.from_dict(team_records, orient='index')

#print(team_df)

count = 0

team_opponents_results = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))

# Populate the team_opponents_results dictionary with win/loss data
for _, row in df_current_season.iterrows():
    home_team, away_team = row['homeTeam'], row['awayTeam']
    home_score, away_score = row['homeScore'], row['awayScore']

    # Determine results for each team
    if home_score > away_score:
        team_opponents_results[home_team][away_team]['wins'] += 1
        team_opponents_results[away_team][home_team]['losses'] += 1
    elif away_score > home_score:
        team_opponents_results[away_team][home_team]['wins'] += 1
        team_opponents_results[home_team][away_team]['losses'] += 1

# At this point, team_opponents_results contains each team's win/loss record against all opponents in the current season

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print the DataFrame
# print(df_current_season)


def point_sort(tied_teams):

    global df_current_season
    # Initialize a dictionary to store points totals for each team in tied_teams
    team_points = {team[0]: 0 for team in tied_teams}

    # Iterate over each game row in the DataFrame to accumulate points
    for index, row in df_current_season.iterrows():
        home_team = row['homeTeam']
        away_team = row['awayTeam']
        home_score = row['homeScore']
        away_score = row['awayScore']

        # Add points only for teams in tied_teams
        if home_team in team_points:
            team_points[home_team] += home_score
        if away_team in team_points:
            team_points[away_team] += away_score

    # Sort tied teams by their total points scored in descending order
    sorted_tied_teams = sorted(tied_teams, key=lambda x: team_points[x[0]], reverse=True)

    # Print the sorted list of tied teams by points scored
    # print("\n\n" + "-" * 40 + "\n\n")
    # print("Sorted by Point Total:\n", team_points)
    # print("\n\n" + "-" * 40 + "\n\n")

    return sorted_tied_teams


def conference_sort(tied_teams):
    # Sort teams by conference win percentage
    sorted_teams = sorted(
        tied_teams,
        key=lambda x: (
            x[1]['current_season'].get('conference_wins', 0) / 
            (x[1]['current_season'].get('conference_wins', 0) + x[1]['current_season'].get('conference_losses', 0))
            if (x[1]['current_season'].get('conference_wins', 0) + x[1]['current_season'].get('conference_losses', 0)) > 0
            else 0
        ),
        reverse=True
    )

    # print("\n\n" + "-" * 40 + "\n\n")
    # print("Sorted by Conference Win Percentage:\n", tied_teams)
    # print("\n\n" + "-" * 40 + "\n\n")


    # Check for ties in conference win percentage
    conference_ties = {}
    for team, stats in sorted_teams:
        conference_win_percentage = (
            stats['current_season'].get('conference_wins', 0) / 
            (stats['current_season'].get('conference_wins', 0) + stats['current_season'].get('conference_losses', 0))
            if (stats['current_season'].get('conference_wins', 0) + stats['current_season'].get('conference_losses', 0)) > 0
            else 0
        )
        if conference_win_percentage not in conference_ties:
            conference_ties[conference_win_percentage] = []
        conference_ties[conference_win_percentage].append((team, stats))

    # Process each group of tied teams with common_games_sort if needed
    final_sorted_teams = []
    for conferece_win_percentage, tied_group in conference_ties.items():
        if len(tied_group) > 1:  # Only process further if there is a tie
            sorted_group = point_sort(tied_group)
            final_sorted_teams.extend(sorted_group)
        else:
            final_sorted_teams.extend(tied_group)



    return tied_teams

def common_games_sort(tied_teams):
    # Create a dictionary to store common game records
    common_games_results = {
        team[0]: {
            'wins': 0,
            'games': 0,
            'current_season': tied_teams[[t[0] for t in tied_teams].index(team[0])][1]['current_season']  # Assuming tied_teams includes this data
        }

    for team in tied_teams
    }
    print("\n\nInto Common Games\n\n")
    # Loop through each pair of tied teams
    for i in range(len(tied_teams)):
        team_a = tied_teams[i][0]
        common_opponents = set()
        
        # Step 1: Identify common opponents for all tied teams
        for team, _ in tied_teams:
            # Get all opponents for the current team
            opponents = set(df_current_season[(df_current_season['homeTeam'] == team)]['awayTeam'].tolist() +
                            df_current_season[(df_current_season['awayTeam'] == team)]['homeTeam'].tolist())
            
            # If this is the first team, initialize common_opponents with its opponents
            if not common_opponents:
                common_opponents = opponents
            else:
                # Find intersection with existing common opponents
                common_opponents.intersection_update(opponents)
        
        # Step 2: Filter games where each team played against any common opponent
        for team, _ in tied_teams:
            relevant_games = df_current_season[((df_current_season['homeTeam'] == team) & (df_current_season['awayTeam'].isin(common_opponents))) |
                                               ((df_current_season['awayTeam'] == team) & (df_current_season['homeTeam'].isin(common_opponents)))]
            
            # Initialize stats if not already present
            if team not in common_games_results:
                common_games_results[team] = {'wins': 0, 'games': 0}
            
            # print("\n\n" + "-" * 40 + "\n\n")
            # print("Sorted by Common Games Win Percentage:\n", relevant_games)
            # print("\n\n" + "-" * 40 + "\n\n")

            # Step 3: Count wins and games for each team against common opponents
            for _, game in relevant_games.iterrows():
                home_team = game['homeTeam']
                away_team = game['awayTeam']
                home_score = game['homeScore']
                away_score = game['awayScore']
                
                # Determine the winner
                if home_score > away_score:
                    winner = home_team
                elif away_score > home_score:
                    winner = away_team
                else:
                    winner = None  # In case of a tie

                # Record win/loss for the current team
                if winner == team:
                    common_games_results[team]['wins'] += 1
                
                # Increment games count for the team
                common_games_results[team]['games'] += 1

    # Calculate win percentages for each team
    for team in common_games_results:
        games = common_games_results[team]['games']
        wins = common_games_results[team]['wins']
        common_games_results[team]['win_percentage'] = wins / games if games > 0 else 0

    # Step 4: Sort tied teams by win percentage
    sorted_teams = sorted(common_games_results.items(), key=lambda x: x[1]['win_percentage'], reverse=True)

    # Step 5: Group teams with tied win percentages
    grouped_by_percentage = {}
    for team, stats in sorted_teams:
        win_percentage = stats['win_percentage']
        if win_percentage not in grouped_by_percentage:
            grouped_by_percentage[win_percentage] = []
        grouped_by_percentage[win_percentage].append((team, stats))

    # print("\n\n" + "-" * 40 + "\n\n")
    # print("Sorted by Common Games Win Percentage:\n", sorted_teams)
    # print("\n\n" + "-" * 40 + "\n\n")

    # Step 6: Process each group; sort by conference if win percentages are tied
    final_sorted_teams = []
    for win_percentage, teams_with_same_percentage in grouped_by_percentage.items():
        if len(teams_with_same_percentage) > 1:
            # Call conference_sort if there is a tie in win percentages
            sorted_group = conference_sort(teams_with_same_percentage)
            final_sorted_teams.extend(sorted_group)
        else:
            final_sorted_teams.extend(teams_with_same_percentage)
    
    return final_sorted_teams




def divisional_sort(tied_teams):
    
    # Sort teams by divisional win percentage
    sorted_teams = sorted(
        tied_teams,
        key=lambda x: (
            x[1]['current_season'].get('division_wins', 0) / 
            (x[1]['current_season'].get('division_wins', 0) + x[1]['current_season'].get('division_losses', 0))
            if (x[1]['current_season'].get('division_wins', 0) + x[1]['current_season'].get('division_losses', 0)) > 0
            else 0
        ),
        reverse=True
    )
    # print("\n\n" + "-" * 40 + "\n\n")
    # print("Sorted by Divisional Win Percentage:\n", sorted_teams)
    # print("\n\n" + "-" * 40 + "\n\n")

    # Check for ties in divisional win percentage
    divisional_ties = {}
    for team, stats in sorted_teams:
        division_win_percentage = (
            stats['current_season'].get('division_wins', 0) / 
            (stats['current_season'].get('division_wins', 0) + stats['current_season'].get('division_losses', 0))
            if (stats['current_season'].get('division_wins', 0) + stats['current_season'].get('division_losses', 0)) > 0
            else 0
        )
        if division_win_percentage not in divisional_ties:
            divisional_ties[division_win_percentage] = []
        divisional_ties[division_win_percentage].append((team, stats))

    # Process each group of tied teams with common_games_sort if needed
    final_sorted_teams = []
    for division_win_percentage, tied_group in divisional_ties.items():
        if len(tied_group) > 1:  # Only process further if there is a tie
            sorted_group = common_games_sort(tied_group)
            final_sorted_teams.extend(sorted_group)
        else:
            final_sorted_teams.extend(tied_group)
    
    return final_sorted_teams


def head_to_head(tied_teams):
    head_to_head_results = {team[0]: {'wins': 0, 'games': 0} for team in tied_teams}
    
    # Loop through each pair of tied teams
    for i in range(len(tied_teams)):
        team_a = tied_teams[i][0]
        for j in range(i + 1, len(tied_teams)):
            team_b = tied_teams[j][0]
            
            # Filter games where team_a or team_b played against each other
            relevant_games = df_current_season[((df_current_season['homeTeam'] == team_a) & (df_current_season['awayTeam'] == team_b)) |
                                               ((df_current_season['homeTeam'] == team_b) & (df_current_season['awayTeam'] == team_a))]
            
            # Count wins for each team in head-to-head matchups
            for _, game in relevant_games.iterrows():
                home_team = game['homeTeam']
                away_team = game['awayTeam']
                home_score = game['homeScore']
                away_score = game['awayScore']
                
                # Determine the winner
                if home_score > away_score:
                    winner = home_team
                elif away_score > home_score:
                    winner = away_team
                else:
                    winner = None  # In case of a tie

                # Record win/loss
                if winner == team_a:
                    head_to_head_results[team_a]['wins'] += 1
                elif winner == team_b:
                    head_to_head_results[team_b]['wins'] += 1
                
                # Increment games count for both teams
                head_to_head_results[team_a]['games'] += 1
                head_to_head_results[team_b]['games'] += 1
    
    # Calculate win percentages for each team
    for team in head_to_head_results:
        games = head_to_head_results[team]['games']
        wins = head_to_head_results[team]['wins']
        head_to_head_results[team]['win_percentage'] = wins / games if games > 0 else 0

    # Sort tied teams by win percentage
    sorted_teams = sorted(head_to_head_results.items(), key=lambda x: x[1]['win_percentage'], reverse=True)

    # print("\n\n" + "-" * 40 + "\n\n")
    # print("Sorted by Head to Head Win Percentage:\n", sorted_teams)
    # print("\n\n" + "-" * 40 + "\n\n")

    # Identify remaining ties by win percentage
    ties = {}
    for team, stats in sorted_teams:
        win_percentage = stats['win_percentage']
        if win_percentage not in ties:
            ties[win_percentage] = []
        ties[win_percentage].append((team, tied_teams[[t[0] for t in tied_teams].index(team)][1]))  # Include full stats here

    # Process each group of tied teams with divisional_sort if needed
    final_sorted_teams = []
    for win_percentage, tied_group in ties.items():
        if len(tied_group) > 1:  # Only process further if there is a tie
            # Run divisional_sort on tied teams with full stats
            sorted_group = divisional_sort(tied_group)
            final_sorted_teams.extend(sorted_group)
        else:
            final_sorted_teams.extend(tied_group)
    
    return final_sorted_teams


def sort_teams(teams):
    # Step 1: Sort teams by wins
    teams_sorted = sorted(teams.items(), key=lambda x: x[1]['current_season']['wins'], reverse=True)
    
    # Step 2: Identify ties by wins
    ties = {}
    for team, stats in teams_sorted:
        wins = stats['current_season']['wins']
        if wins not in ties:
            ties[wins] = []
        ties[wins].append((team, stats))

    # Step 3: Process each group of tied teams with head_to_head for further tiebreaking
    final_sorted_teams = []
    for win_count, tied_teams in ties.items():
        if len(tied_teams) > 1:  # Only process if there is a tie
            sorted_group = head_to_head(tied_teams)
            final_sorted_teams.extend(sorted_group)
        else:
            final_sorted_teams.extend(tied_teams)
    
    # Convert back to dictionary format for the final sorted dictionary
    final_sorted_dict = dict((team, stats) for team, stats in final_sorted_teams)
    
    # Print the sorted dictionary with ties resolved
    # print(final_sorted_dict)
    return final_sorted_dict



def merge_team_records(team, simulated_records, division_records, conference_records):
    return {
        'current_season': {
            'wins': simulated_records[team]['wins'],
            'losses': simulated_records[team]['games'] - simulated_records[team]['wins'],
            'games': simulated_records[team]['games'],
            'division_wins': division_records[team]['division_wins'],
            'division_losses': division_records[team]['division_losses'],
            'conference_wins': conference_records[team]['conference_wins'],
            'conference_losses': conference_records[team]['conference_losses'],
            'common_games': {'wins': 0, 'losses': 0, 'games': 0}  # Placeholder; update as needed
        }
    }




def simulate_season(team_records, future_games, iterations=1000):

    playoff_counts = {team: 0 for team in team_records}
    first_counts = {team: 0 for team in team_records}

    # Initialize playoff and Super Bowl counts
    super_bowl_wins = {team: 0 for team in team_records}

    def simulate_playoff_game(team_a, team_b, win_pct_a, win_pct_b):
        """Simulate a playoff game based on win percentages."""
        total_prob = win_pct_a + win_pct_b
        adjusted_prob_a = win_pct_a / total_prob
        return team_a if np.random.rand() < adjusted_prob_a else team_b

    for _ in range(iterations):

        simulated_common_opponents = {team: {} for team in team_records.keys()}
        simulated_records = {team: {'wins': record['current_season']['wins'], 'games': record['current_season']['games']} for team, record in team_records.items()}
        division_records = {team: {'division_wins': record['current_season']['division_wins'], 'division_losses': record['current_season']['division_losses']} for team, record in team_records.items()}
        conference_records = {team: {'conference_wins': record['current_season']['conference_wins'], 'conference_losses': record['current_season']['conference_losses']} for team, record in team_records.items()}

        for index, row in future_games.iterrows():
            home_team = row['homeTeam']
            away_team = row['awayTeam']
            home_win_pct = team_records[home_team]['win_percentage']
            away_win_pct = team_records[away_team]['win_percentage']
            
            # Generate the minimum home win percentage
            total_prob = home_win_pct + away_win_pct
            adjusted_home_win_prob = home_win_pct / total_prob
            random_number = np.random.rand()
            
                # Initialize head-to-head records if not already set
            if (home_team, away_team) not in head_to_head_results:
                head_to_head_results[(home_team, away_team)] = {'wins': 0, 'losses': 0}
                head_to_head_results[(away_team, home_team)] = {'wins': 0, 'losses': 0}
                
            # Check and log the win or loss
            if random_number < adjusted_home_win_prob:
                simulated_records[home_team]['wins'] += 1
                head_to_head_results[(home_team, away_team)]['wins'] += 1
                head_to_head_results[(away_team, home_team)]['losses'] += 1
                simulated_common_opponents[home_team].setdefault(away_team, {'win_pct': 0, 'games': 0})
                simulated_common_opponents[home_team][away_team]['win_pct'] += 1
                simulated_common_opponents[home_team][away_team]['games'] += 1
            else:
                simulated_records[away_team]['wins'] += 1
                head_to_head_results[(away_team, home_team)]['wins'] += 1
                head_to_head_results[(home_team, away_team)]['losses'] += 1
                simulated_common_opponents[away_team].setdefault(home_team, {'win_pct': 0, 'games': 0})
                simulated_common_opponents[away_team][home_team]['win_pct'] += 1
                simulated_common_opponents[away_team][home_team]['games'] += 1
            

            # Check for division game
            if row['home_team_division'] == row['away_team_division']:
                if random_number < adjusted_home_win_prob:
                    division_records[home_team]['division_wins'] += 1
                    division_records[away_team]['division_losses'] += 1
                else:
                    division_records[away_team]['division_wins'] += 1
                    division_records[home_team]['division_losses'] += 1
            
            # Check for conference game
            if row['home_team_conference'] == row['away_team_conference']:
                if random_number < adjusted_home_win_prob:
                    conference_records[home_team]['conference_wins'] += 1
                    conference_records[away_team]['conference_losses'] += 1
                else:
                    conference_records[away_team]['conference_wins'] += 1
                    conference_records[home_team]['conference_losses'] += 1

            
            simulated_records[home_team]['games'] += 1
            simulated_records[away_team]['games'] += 1

        # Complete AFC and NFC records with merged data
        afc_team_records = {team: merge_team_records(team, simulated_records, division_records, conference_records) for team in afc_teams_list if team in simulated_records}
        nfc_team_records = {team: merge_team_records(team, simulated_records, division_records, conference_records) for team in nfc_teams_list if team in simulated_records}



        """
        # Print sorted AFC teams with their sort keys for debugging
        print("AFC Sorted Teams and their Rankings:")
        for team, record in afc_sorted_teams:
            sort_key_values = calculate_sort_key(team, record, afc_team_records, head_to_head_results, team_opponents_results)
            print(f"{team}: Sort Key = {sort_key_values}")

        print("\n" + "-" * 40 + "\n")  # Separator for readability

        # Print sorted NFC teams with their sort keys for debugging
        print("NFC Sorted Teams and their Rankings:")
        for team, record in nfc_sorted_teams:
            sort_key_values = calculate_sort_key(team, record, nfc_team_records, head_to_head_results, team_opponents_results)
            print(f"{team}: Sort Key = {sort_key_values}")

        print("\n" + "-" * 40 + "\n")
        """

        # Convert sorted dictionaries to lists of tuples for AFC and NFC
        afc_sorted_teams = list(sort_teams(afc_team_records).items())
        nfc_sorted_teams = list(sort_teams(nfc_team_records).items())

        # Get the top 7 teams for each conference
        afc_teams = afc_sorted_teams[:7]  # Top 7 teams in AFC
        nfc_teams = nfc_sorted_teams[:7]  # Top 7 teams in NFC

        # Get the first-ranked teams in each conference
        first_afc_teams = afc_sorted_teams[:1]
        first_nfc_teams = nfc_sorted_teams[:1]
        
        # Update counts for first place teams and playoff teams
        for team, _ in first_afc_teams + first_nfc_teams:
            first_counts[team] += 1

        for team, _ in afc_teams + nfc_teams:
            playoff_counts[team] += 1
        
        # Calculate playoff chances
        firstSeed = {team: first_counts[team] / iterations for team in first_counts}

        # print("\n\n" + "-" * 40 + "\n\n")
        # print("Advancing Teams:\n", afc_teams)
        # print("\n\n" + "-" * 40 + "\n\n")


            # Simulate the AFC and NFC playoffs
        for conference, teams in [('AFC', afc_teams), ('NFC', nfc_teams)]:
            advancing_teams = [team[0] for team in teams]

            # print("\n\n" + "-" * 40 + "\n\n")
            # print("Advancing Teams:\n", advancing_teams)
            # print("\n\n" + "-" * 40 + "\n\n")

            # First Round
            round_1_winners = [
                simulate_playoff_game(advancing_teams[1], advancing_teams[6], team_records[advancing_teams[1]]['win_percentage'], team_records[advancing_teams[6]]['win_percentage']),
                simulate_playoff_game(advancing_teams[2], advancing_teams[5], team_records[advancing_teams[2]]['win_percentage'], team_records[advancing_teams[5]]['win_percentage']),
                simulate_playoff_game(advancing_teams[3], advancing_teams[4], team_records[advancing_teams[3]]['win_percentage'], team_records[advancing_teams[4]]['win_percentage'])
            ]

             
            # Second Round
            highest_seeded_winner = advancing_teams[0]
            round_2_teams = [highest_seeded_winner] + sorted(round_1_winners, key=lambda team: advancing_teams.index(team))
            round_2_winners = [
                simulate_playoff_game(round_2_teams[0], round_2_teams[3], team_records[round_2_teams[0]]['win_percentage'], team_records[round_2_teams[3]]['win_percentage']),
                simulate_playoff_game(round_2_teams[1], round_2_teams[2], team_records[round_2_teams[1]]['win_percentage'], team_records[round_2_teams[2]]['win_percentage'])
            ]
            
            # Conference Championship
            conference_champion = simulate_playoff_game(round_2_winners[0], round_2_winners[1], team_records[round_2_winners[0]]['win_percentage'], team_records[round_2_winners[1]]['win_percentage'])
            
            # Track Super Bowl contenders
            # playoff_counts[conference_champion] += 1
            if conference == 'AFC':
                afc_champion = conference_champion
            else:
                nfc_champion = conference_champion

        # Simulate the Super Bowl
        super_bowl_winner = simulate_playoff_game(afc_champion, nfc_champion, team_records[afc_champion]['win_percentage'], team_records[nfc_champion]['win_percentage'])
        super_bowl_wins[super_bowl_winner] += 1
    
    # Calculate playoff chances
    playoff_chances = {team: playoff_counts[team] / iterations for team in playoff_counts}
    super_bowl_percentages = {team: (super_bowl_wins[team] / iterations) * 100 for team in super_bowl_wins}

    return playoff_chances, firstSeed, super_bowl_percentages





# Define AFC and NFC teams lists
afc_teams_list = ['Patriots', 'Bills', 'Dolphins', 'Jets', 'Ravens', 'Steelers', 'Browns', 'Bengals', 'Texans', 'Colts', 'Jaguars', 'Titans', 'Broncos', 'Chiefs', 'Raiders', 'Chargers']

nfc_teams_list = ['Cowboys', 'Giants', 'Eagles', 'Commanders', 'Packers', 'Bears', 'Vikings', 'Lions', 'Falcons', 'Panthers', 'Saints', 'Buccaneers', 'Cardinals', 'Rams', '49ers', 'Seahawks']

# Simulate season and calculate playoff chances
playoff_chances, firstSeed, super_bowl_chances = simulate_season(team_records, future_games)
super_bowl_chances = dict(sorted(super_bowl_chances.items(), key=lambda item: item[1], reverse=True))

# Sort playoff chances in descending order
sorted_playoff_chances = dict(sorted(playoff_chances.items(), key=lambda item: item[1], reverse=True))
firstSeed = dict(sorted(firstSeed.items(), key=lambda item: item[1], reverse=True))

# Filter for AFC teams
afc_playoff_chances = {team: chance for team, chance in sorted_playoff_chances.items() if team in afc_teams_list}
afc_firstSeed = {team: chance for team, chance in firstSeed.items() if team in afc_teams_list}

# Filter for NFC teams
nfc_playoff_chances = {team: chance for team, chance in sorted_playoff_chances.items() if team in nfc_teams_list}
nfc_firstSeed = {team: chance for team, chance in firstSeed.items() if team in nfc_teams_list}

# Print playoff chances for AFC teams
print("\n\nAFC Teams Playoff Chances:")
for team, chance in afc_playoff_chances.items():
    print(f"{team}: {chance * 100:.2f}%")

# Print playoff chances for NFC teams
print("\n\nNFC Teams Playoff Chances:")
for team, chance in nfc_playoff_chances.items():
    print(f"{team}: {chance * 100:.2f}%")

# Print playoff chances for AFC teams
print("\n\nAFC Teams 1st Seed Chances:")
for team, chance in afc_firstSeed.items():
    print(f"{team}: {chance * 100:.2f}%")

# Print playoff chances for NFC teams
print("\n\nNFC Teams 1st Seed Chances:")
for team, chance in nfc_firstSeed.items():
    print(f"{team}: {chance * 100:.2f}%")

print("\nSuper Bowl Win Percentages:")
for team, pct in super_bowl_chances.items():
    print(f"{team}: {pct:.2f}%")

# Create a DataFrame for team records and playoff chances
team_df = pd.DataFrame.from_dict(team_records, orient='index')
playoff_chances_df = pd.DataFrame.from_dict(sorted_playoff_chances, orient='index', columns=['playoff_chance'])

# Merge the dataframes
result_df = team_df.merge(playoff_chances_df, left_index=True, right_index=True)

#print(result_df)

print("\n\n--- %s seconds ---" % (time.time() - start_time))

# Define function to convert percentage to American odds
def percentage_to_american_odds(percentage):
    if percentage == 100.0:
        return '-Infinity'  # Represents 100% certainty (no payout in betting terms)
    elif percentage == 0.0:
        return 'No odds'  # Represents 0% chance
    else:
        decimal_odds = 100 / percentage
        if decimal_odds >= 2:
            return f"+{int((decimal_odds - 1) * 100)/2}"
        else:
            return f"{int(-100 / (decimal_odds - 1))/2}"

# Convert all percentages to American odds
afc_playoff_odds = {team: percentage_to_american_odds(pct) for team, pct in afc_playoff_chances.items()}
nfc_playoff_odds = {team: percentage_to_american_odds(pct) for team, pct in nfc_playoff_chances.items()}
afc_seed_odds = {team: percentage_to_american_odds(pct) for team, pct in afc_firstSeed.items()}
nfc_seed_odds = {team: percentage_to_american_odds(pct) for team, pct in nfc_firstSeed.items()}
super_bowl_odds = {team: percentage_to_american_odds(pct) for team, pct in super_bowl_chances.items()}

# Create dataframes for organized display
afc_playoff_odds_df = pd.DataFrame(list(afc_playoff_odds.items()), columns=['AFC Team', 'Playoff Odds'])
nfc_playoff_odds_df = pd.DataFrame(list(nfc_playoff_odds.items()), columns=['NFC Team', 'Playoff Odds'])
afc_seed_odds_df = pd.DataFrame(list(afc_seed_odds.items()), columns=['AFC Team', '1st Seed Odds'])
nfc_seed_odds_df = pd.DataFrame(list(nfc_seed_odds.items()), columns=['NFC Team', '1st Seed Odds'])
super_bowl_odds_df = pd.DataFrame(list(super_bowl_odds.items()), columns=['Team', 'Super Bowl Odds'])


print("\nSuper Bowl Win Percentages:")
for index, row in super_bowl_odds_df.iterrows():
    try:
        # Attempt to convert to float and round to the nearest 50
        rounded_odds = round(float(row['Super Bowl Odds']) / 50) * 50
        print(f"{row['Team']}: +{rounded_odds}")
    except ValueError:
        # If conversion fails, print the original string
        print(f"{row['Team']}: {row['Super Bowl Odds']}")
