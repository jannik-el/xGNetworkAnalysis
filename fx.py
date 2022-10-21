## This file is for creating functions to shorten down on boilerplate code that we use often

# libs
import json
import numpy as np
import pandas as pd
import networkx as nx
import math
import seaborn as sns
import matplotlib.pyplot as plt
from mplsoccer import Pitch, VerticalPitch
from scipy.ndimage import gaussian_filter
from statsbombpy import sb

# funcs
def PullSBData(competition_name, season):
    """
    Takes: Competition Name
    Returns: comp_id, season_id
    """
    competitions = sb.competitions()
    comp_id = int(competitions[competitions['competition_name']==competition_name]['competition_id'])
    season_id = int(competitions[competitions['competition_name']==competition_name]['season_id'])
    return comp_id, season_id

def ReturnMatchIDs(comp_id, season_id):
    """Returns MatchIDs and Teams (for streamlit)"""
    matches = sb.matches(competition_id = comp_id, season_id = season_id)
    matchdict = {f'{list(matches["home_team"])[i]} vs {list(matches["away_team"])[i]}': list(matches["match_id"])[i] for i in range(len(list(matches["match_id"])))}
    return matchdict

def ReturnCompetitions():
    """Returns Competitions in Statsbomb Dataset"""
    competitions = sb.competitions()
    return list(competitions["competition_name"].unique())

def ReturnSeasons(competition_name):
    """Returns Seasons in Statsbomb Dataset"""
    competitions = sb.competitions()
    comp_id = competition_data[competition_data['competition_name']==competition_name]['competition_id'].iloc[0]
    return list(comp_id['season_name'].unique())

def ReturnScoreInfo(comp_id, season_id, match_id):
    """
    Returns: [MatchID, DateTime of Match Kickoff, Teams [Home, Away], Scores [Home, Away]]
    """
    scores_df = sb.matches(comp_id, season_id)
    m_id = scores_df[scores_df["match_id"] == match_id]
    date_time = str(m_id["match_date"]).split()[1] + " " + str(m_id["kick_off"]).split()[1]
    teams = [str(m_id["home_team"]).split()[1], str(m_id["away_team"]).split()[1]]
    scores = [int(m_id["home_score"]), int(m_id["away_score"])]
    return [match_id, date_time, teams, scores]

def CreateEventsDF(comp_id, season_id, match_id, hometeam):
    """
    Takes: comp_id, season_id, match_id, hometeam (str)
    Returns: events (a pandas DF)
    """
    matches = sb.matches(competition_id = comp_id, season_id = season_id)
    matches = matches[matches['home_team'] == hometeam]
    events = sb.events(match_id)
    print("Number of matches: {}".format(len(matches)))
    print("Number of events for a sample match: {}".format(len(events)))
    return events

def CreatePassDF(events, hometeam):
    """
    Takes: events (created by CreateEventsDF), hometeam (str)
    Returns: pass_df (a DF with all pass information of the home team)
    """
    events = events[['minute', 'second', 'team', 'location', 'period', 'type', 'pass_outcome', 'player', 'position', 'pass_end_location', 'pass_recipient']]
    events = events.rename(columns={'pass_recipient': 'recipient'})
    team_df = events[events['team'] == hometeam]
    pass_df = team_df[team_df['type'] == 'Pass']
    #taking all the values that are null because that means that it was a successfull pass
    pass_df = pass_df[pass_df['pass_outcome'].isnull()] 
    pass_df['passer'] = pass_df['player']
    return pass_df

# # not really sure what this function does and why
# def GetPlayers(events):
#     """
#     Takes: events (created by CreateEventsDF)
#     returns: players_x (a DF of all players in a team)
#     """
#     tact = events[events['tactics'].isnull() == False]
#     tact = tact[['tactics', 'team', 'type']]
#     team_x = list(tact['team'])[0]
#     tact_x = tact[tact['team'] == team_x]['tactics']
#     dict_x = tact_x[0]['lineup']
#     lineup_x = pd.DataFrame.from_dict(dict_x)
#     players_x = {}
#     for i in range(len(lineup_x)):
#         key = lineup_x.player[i]['name']
#         val = lineup_x.jersey_number[i]
#         players_x[key] = str(val)
#     return players_x

def GetPlayers(match_id, team):
    lineup = sb.lineups(match_id=match_id)[team]
    lineup['player_nickname'].fillna(lineup['player_name'], inplace=True)
    players = list(lineup['player_nickname'])
    return dict(zip(list(lineup['player_nickname']), list(lineup['jersey_number'])))



def ReturnSubstitutionMinutes(events, team):
    """
    Takes: events, team (str)
    Returns: List of Lists with minutes and Seconds of Substitution
    """
    subs = events[events['type']=='Substitution']
    subs = subs[subs['team']==team]
    mapit = lambda n, m : [n, m]
    return list(map(mapit, list(subs['minute']), list(subs['second'])))

def ReturnAvgPositionsDF(pass_df):
    """
    Takes: pass_df
    Returns: pass_bet (DF of passes between), avg_loc(DF of average locations)
    """
    pass_loc = pass_df['location']
    pass_loc = pd.DataFrame(pass_loc.to_list(), columns=['x', 'y'])
    pass_end_loc = pass_df['pass_end_location']
    pass_end_loc = pd.DataFrame(pass_end_loc.to_list(), columns=['end_x', 'end_y'])
    
    pass_df = pass_df.reset_index()
    pass_df['x'] = pass_loc['x']
    pass_df['y'] = pass_loc['y']
    pass_df['end_x'] = pass_end_loc['end_x']
    pass_df['end_y'] = pass_end_loc['end_y']
    pass_df = pass_df.drop(columns=['location', 'pass_end_location', 'pass_outcome'])

    avg_loc = pass_df.groupby('passer').agg({'x':['mean'], 'y': ['mean', 'count']})
    avg_loc.columns=['x', 'y', 'count']

    pass_bet = pass_df.groupby(['passer', 'recipient']).index.count().reset_index()
    pass_bet.rename({'index':'pass_count'}, axis='columns', inplace=True)

    pass_bet = pass_bet.merge(avg_loc, left_on = 'passer', right_index=True)
    pass_bet = pass_bet.merge(avg_loc, left_on = 'recipient', right_index=True, suffixes=['', '_end'])
    return pass_bet, avg_loc

# TODO add a title to this plot
def PlotPitch(pass_bet, avg_loc):
    """
    Takes: pass_bet, avg_loc
    Returns: plt.show() of the average player positions
    """
    pitch = Pitch(pitch_color='grass', pitch_type='statsbomb',line_color='white',stripe=True, goal_type='box', label=True, axis=True, tick=True)
    fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)

    arrows = pitch.arrows(pass_bet.x, pass_bet.y, pass_bet.x_end, pass_bet.y_end, ax=ax, width=5,
                        headwidth=3, color='white', zorder=1, alpha=0.5)
    nodes = pitch.scatter(avg_loc.x, avg_loc.y, s=400, color='blue', edgecolors='black', linewidth=2.5, alpha=1, zorder=1, ax=ax)
    return plt.show()

def ReturnNXPassNetwork(pass_bet):
    """
    Takes: pass_bet
    Returns: G (NX Pass Network)
    """
    graph = pass_bet[['passer', 'recipient', 'pass_count']]
    L = graph.apply(tuple, axis=1).tolist()
    
    G = nx.DiGraph()
    G.add_weighted_edges_from(L)
    return G

def PlotPlayerDegrees(G):
    """
    Takes: G (NX Pass Network)
    Returns: plt.show() of different 
    """
    

    # degrees
    dic = dict(nx.degree(G))
    player = dic.keys()
    degrees = dic.values()
    degree_fr = pd.DataFrame({'player':player, 'degrees':degrees})
    deg_ordered = degree_fr.sort_values(by='degrees')
    deg_x_range = range(len(degree_fr.index))
    deg_y_range = range(math.ceil(max(degree_fr.degrees)))

    # indegrees
    dic = dict(G.in_degree())
    player = dic.keys()
    in_degrees = dic.values()
    in_degree_fr = pd.DataFrame({'player':player, 'in_degrees':in_degrees})
    indeg_ordered = in_degree_fr.sort_values(by = 'in_degrees')
    indeg_x_range = range(len(in_degree_fr.index))
    indeg_y_range = range(math.ceil(max(in_degree_fr.in_degrees)))

    # outdegrees
    dic = dict(G.out_degree())
    player = dic.keys()
    out_degrees = dic.values()
    out_degree = pd.DataFrame({'player':player, 'out_degrees':out_degrees})
    outdeg_ordered = out_degree.sort_values(by = 'out_degrees')
    outdeg_x_range = range(len(out_degree.index))
    outdeg_y_range = range(math.ceil(max(out_degree.out_degrees)))

    # plotting
    fig, ax = plt.subplots(1, 3, figsize=[25, 8])
    ax[0].stem(deg_ordered['degrees'])
    ax[0].set_xticks(deg_x_range, deg_ordered['player'], rotation=90)
    ax[0].set_yticks(deg_y_range)
    ax[0].set_ylabel("degree (total number of passes played)")
    ax[0].set_title("Successful passes (degrees) of each player (vertex)", size=10)

    ax[1].stem(indeg_ordered['in_degrees'])
    ax[1].set_xticks(indeg_x_range, indeg_ordered['player'], rotation=90)
    ax[1].set_yticks(indeg_y_range)
    ax[1].set_ylabel("in degree (total number of passes received)")
    ax[1].set_title("Successful passes received (indegrees) for each player (vertex)", size=10)

    ax[2].stem(outdeg_ordered['out_degrees'])
    ax[2].set_xticks(outdeg_x_range, outdeg_ordered['player'], rotation=90)
    ax[2].set_yticks(outdeg_y_range)
    ax[2].set_ylabel("out degree (total number of passes given)")
    ax[2].set_title("Successful passes given (outdegrees) by each player (vertex)", size=10)

    return plt.show()

def CreatexGDF(match_id):
    """
    Takes: Match ID
    # Returns: list of xG data + team names in that order:
                [Team 1 name, Team 1 xG minutes, Team 1 Cumulative xG, Team 2 name, Team 2 xG minutes, Team 2 Cumulative xG]
    """

    events = sb.events(match_id)
    df_shots = events[['location', 'minute', 'player', 'team', 'shot_outcome', 'shot_statsbomb_xg', 'shot_technique', 'shot_type']]
    df_shots = df_shots[df_shots['shot_outcome'].isnull()==False].reset_index()

    teams = list(df_shots['team'].unique())
    team1 = teams[0]
    team2 = teams[1]

    df_shots_team1 = df_shots[df_shots['team'] == team1].reset_index()
    df_shots_team2 = df_shots[df_shots['team'] == team2].reset_index()

    shots_team1_xg = df_shots_team1['shot_statsbomb_xg'].tolist()
    shots_team1_xg_minute = df_shots_team1['minute'].tolist()

    shots_team2_xg = df_shots_team2['shot_statsbomb_xg'].tolist()
    shots_team2_xg_minute = df_shots_team2['minute'].tolist()
    
    team1_xg_cumu = (np.cumsum(shots_team1_xg)).tolist()
    team2_xg_cumu = (np.cumsum(shots_team2_xg)).tolist()

    shots_team1_xg_minute = [0] + shots_team1_xg_minute + [int((shots_team1_xg_minute[-1] + 1))]
    shots_team2_xg_minute = [0] + shots_team2_xg_minute + [int((shots_team1_xg_minute[-1] + 1))]
    team1_xg_cumu = [0] + team1_xg_cumu + [(team1_xg_cumu[-1])]
    team2_xg_cumu = [0] + team2_xg_cumu + [(team1_xg_cumu[-1])]

    return [team1, shots_team1_xg_minute, team1_xg_cumu, team2, shots_team2_xg_minute, team2_xg_cumu]


def PlotxG(xg_data):
    team1, shots_team1_xg_minute, team1_xg_cumu, team2, shots_team2_xg_minute, team2_xg_cumu = xg_data

    fig, ax = plt.subplots(figsize=(13.5, 8))

    plt.xticks(range(0, 105, 15))
    plt.xlabel('Time in Minutes')
    plt.ylabel('xG')

    ax.step(shots_team1_xg_minute, team1_xg_cumu, where='post', color='black', label = team1, linewidth=6)
    ax.step(shots_team2_xg_minute, team2_xg_cumu, where='post', color='red', label = team2, linewidth=6)
    ax.legend(borderpad=1, markerscale=0.5, labelspacing=1.5, fontsize=10)
    ax.title.set_text(f"The xG Progress Chart Between {team1} against {team2}")
    return plt.show()