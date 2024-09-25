import pandas as pd
import numpy as np


# possible improvements: normalize the data by their game played
def create_relationship_matrix(df, target_feature):
    feature_map = {
        'Goals': ('HomeScore', 'AwayScore'),
        'Possession': ('HomePossession', 'AwayPossession'),
        'Fouls': ('HomeFouls', 'AwayFouls'),
        'SuccessfulPassesPct': ('HomeSuccessfulPassesPct', 'AwaySuccessfulPassesPct'),
        'RedCards': ('HomeRedCards', 'AwayRedCards'),
    }
    
    if target_feature not in feature_map:
        raise ValueError(f"Target feature '{target_feature}' is not supported. Choose from 'Goals', 'Possession', or 'Fouls'.")

    home_col, away_col = feature_map[target_feature]
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    matrix = pd.DataFrame(np.zeros((len(teams), len(teams))), index=teams, columns=teams)
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        feature_diff = row[home_col] - row[away_col]
        matrix.loc[home_team, away_team] += feature_diff
        matrix.loc[away_team, home_team] += -feature_diff
    return matrix

# Example usage:
# goal_matrix = create_relationship_matrix(matches_up_to_date, 'Goals')
# possession_matrix = create_relationship_matrix(matches_up_to_date, 'Possession')
# fouls_matrix = create_relationship_matrix(matches_up_to_date, 'Fouls')
