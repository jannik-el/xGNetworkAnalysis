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
from dateutil import parser
import statistics
from itertools import chain
from tqdm import tqdm


# funcs
def ReturnComp_Id(competition_name):
    """
    Takes: Competition Name (str)
    Returns: comp_id
    """
    competitions = sb.competitions()
    comp_id = list(competitions[competitions['competition_name']==competition_name]['competition_id'])[0]
    return comp_id

def ReturnMatchID_Dict(comp_id, season_id):
    """
    Takes: Competition ID and Season ID (both int)
    Returns: MatchIDs and Teams (for streamlit)
    """
    matches = sb.matches(competition_id = comp_id, season_id = season_id)
    matchdict = {f'{list(matches["home_team"])[i]} vs {list(matches["away_team"])[i]}': list(matches["match_id"])[i] for i in range(len(list(matches["match_id"])))}
    return matchdict

def ReturnMatchID_List():
    """
    Returns a list of every single matchID
    """
    seasonid_dict = {}
    for c in ReturnCompetitions():
        season_ids = [ReturnSeason_Id(i) for i in ReturnSeasons(c)]
        comp_id = ReturnComp_Id(c)
        seasonid_dict.setdefault(comp_id, season_ids)
    # seasonid 16, comp 76 was causing some problems, idk y
    seasonid_dict[16].pop(-1)
    
    match_ids = []
    for x, i in enumerate(tqdm(seasonid_dict.items())):
        for j in i[1]:
            matchidcol = list(sb.matches(i[0], j)["match_id"].values)
            match_ids.append(matchidcol)

    match_ids = list(chain(*match_ids))

    return match_ids

def ReturnCompetitions():
    """
    Takes: nothing : - )
    Returns: names of Competitions in Statsbomb Dataset
    """
    competitions = sb.competitions()
    return list(competitions["competition_name"].unique())

def ReturnSeasons(competition_name):
    """
    Takes: Competition name (str)
    Returns: Seasons in Statsbomb Dataset
    """
    competitions = sb.competitions()
    competition_data = competitions[competitions['competition_name']==competition_name]
    return list(competition_data['season_name'].unique())

def ReturnSeason_Id(season):
    """
    Takes: Season name (str)
    Returns: Season ID
    """
    competition_data = sb.competitions()
    competition_data = competition_data[competition_data['season_name'] == season]
    season_id = competition_data['season_id'].iloc[0] #season_id for the season_name
    return season_id

def ReturnScoreInfo(comp_id, season_id, match_id):
    """
    Takes: Competition ID, season ID and match ID (all int)
    Returns: [MatchID, DateTime of Match Kickoff, Teams [Home, Away], Scores [Home, Away]]
    """
    scores_df = sb.matches(comp_id, season_id)
    m_id = scores_df[scores_df["match_id"] == match_id]
    date_time = str(m_id["match_date"]).split()[1] + " " + str(m_id["kick_off"]).split()[1]
    teams = [str(m_id["home_team"]).split()[1], str(m_id["away_team"]).split()[1]]
    scores = [int(m_id["home_score"]), int(m_id["away_score"])]
    return [match_id, date_time, teams, scores]

def CreateEventsDF(match_id):
    """
    Takes: match_id (int)
    Returns: events (a pandas DF)
    """
    return sb.events(match_id)

def CreatePassDF(events, hometeam):
    """
    Takes: events (created by CreateEventsDF), hometeam (str)
    Returns: pass_df (a DF with all pass information of the home team)
    """
    events = events[['minute', 'second', 'team', 'location', 'period', 'type', 'pass_outcome', 'player', 'pass_end_location', 'pass_recipient']]
    events = events.rename(columns={'pass_recipient': 'recipient'})
    team_df = events[events['team'] == hometeam]
    pass_df = team_df[team_df['type'] == 'Pass']
    #taking all the values that are null because that means that it was a successfull pass
    pass_df = pass_df[pass_df['pass_outcome'].isnull()] 
    pass_df['passer'] = pass_df['player']
    return pass_df

def GetPlayers(match_id, team):
    """
    Takes: match ID (int), team (str)
    """
    lineup = sb.lineups(match_id=match_id)[team]
    lineup['player_nickname'].fillna(lineup['player_name'], inplace=True)
    players = list(lineup['player_nickname'])
    return dict(zip(list(lineup['player_nickname']), list(lineup['jersey_number'])))

def get_team_names(comp_id = None,season_id = None, match_id = None):
    """
    Takes: either comp_id and season_id or match_id
    Returns: list of team names for eligible games
    """
    if ((not comp_id or not season_id) and not match_id):
        print('this just got exectued')
        raise RuntimeError('Wrong input: either both comp_id and season_id or just match_id')
    if match_id:
        return list(sb.events(match_id)['team'].unique())
    else:
        match_ids_dict = fx.ReturnMatchIDs(comp_id, season_id)
        team_names = [s.split(' vs ') for s in match_ids_dict.keys()]
        if len(team_names) == 1 : return team_names[0]
        return team_names

def ReturnSubstitutionMinutes(events, team):
    """
    Takes: events (from CreateEventsDF), team (str)
    Returns: List of Lists with minutes and Seconds of Substitution
    """
    subs = events[events['type']=='Substitution']
    subs = subs[subs['team']==team]
    mapit = lambda n, m : [n, m]
    return list(map(mapit, list(subs['minute']), list(subs['second'])))

def ReturnAvgPositionsDF(pass_df):
    """
    Takes: pass_df (from CreatePassDF)
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
    Takes: pass_bet (from ReturnAvgPositionsDF)
    Returns: G (NX Pass Network)
    """
    graph = pass_bet[['passer', 'recipient', 'pass_count']]
    L = graph.apply(tuple, axis=1).tolist()
    
    G = nx.DiGraph()
    G.add_weighted_edges_from(L)
    return G

def PlotPlayerDegrees(G):
    """
    Takes: G (NX Pass Network from ReturnNXPassNetwork)
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
    Takes: Match ID (int)
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
    shots_team2_xg_minute = [0] + shots_team2_xg_minute + [int((shots_team2_xg_minute[-1] + 1))]
    team1_xg_cumu = [0] + team1_xg_cumu + [(team1_xg_cumu[-1])]
    team2_xg_cumu = [0] + team2_xg_cumu + [(team2_xg_cumu[-1])]

    return [team1, shots_team1_xg_minute, team1_xg_cumu, team2, shots_team2_xg_minute, team2_xg_cumu]


def PlotxG(xg_data):
    """
    Takes: xg_data(team1, shots_team1_xg_minute, 
           team1_xg_cumu, team2, shots_team2_xg_minute, team2_xg_cumu)
    Returns: Plot of the progress of xG between two teams
    """
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

def SplitEvents(events_df, min_event_time):
    """
    Takes: Events_df, min_event_time
    Returns: List of events_df for splits in game
    """
    events_df = events_df[['id', 'period', 'timestamp', 'minute', 'second', 
                        'team', 'player','location', 'type', 
                        'pass_outcome', 'shot_outcome', 'shot_statsbomb_xg', 
                        'pass_end_location', 'pass_recipient']]

    events_df = events_df.sort_values(['period','timestamp'], ascending=[True, True])
    events_df=events_df.reset_index()
    events_df = events_df.drop("index", axis=1)
    events_df = events_df.reset_index()

    half_ends_lst = dict(zip(events_df[events_df['type']=='Half End']['period'], events_df[events_df['type']=='Half End']['timestamp']))


    goal = events_df[events_df['shot_outcome']=='Goal']
    goals_index_lst = list(events_df[events_df['shot_outcome']=='Goal'].index)
    subs = events_df[events_df['type']=='Substitution']
    subs_index_lst = list(events_df[events_df['type']=='Substitution'].index)

    period_index_lst = []
    for i in events_df['period'].unique():
        idx = events_df[events_df['period'] == i].index[0]
        period_index_lst.append(idx)

    goals_index_dct = {x+1:"goal" for x in goals_index_lst}
    subs_index_dct = {x+1:"sub" for x in subs_index_lst}
    breaks_index_dct = {x+1:"break" for x in period_index_lst}

    splitting_events = {**goals_index_dct, **subs_index_dct, **breaks_index_dct} # very pretty line
    splitting_sorted = sorted(list(splitting_events.keys()))

    splitting_df = pd.DataFrame.from_dict(splitting_events, orient='index', columns=["event"])
    splitting_df = splitting_df.reset_index()
    
    clean_splitting_df=splitting_df.merge(events_df, on='index')
    clean_splitting_df = clean_splitting_df.sort_values("index")

    clean_splitting_df['timestamp'] = clean_splitting_df['timestamp'].apply(lambda x: parser.parse(x))

    clean_splitting_df['time_delta'] =  clean_splitting_df['timestamp'] - clean_splitting_df['timestamp'].shift(1, fill_value=0)

    clean_splitting_df['time_delta'] = clean_splitting_df['time_delta'].astype('timedelta64[m]')
    clean_splitting_df = clean_splitting_df.reset_index()

    clean_splitting_df['half_end'] = clean_splitting_df['period'].map(half_ends_lst)
    clean_splitting_df['half_end'] = clean_splitting_df['half_end'].apply(lambda x: parser.parse(x))
    clean_splitting_df['diff_to_half_end'] = clean_splitting_df['half_end'] - clean_splitting_df['timestamp']
    clean_splitting_df['diff_to_half_end'] = clean_splitting_df['diff_to_half_end'].astype('timedelta64[m]')
    clean_splitting_df = clean_splitting_df[clean_splitting_df['diff_to_half_end'] > min_event_time]

    clean_splitting_df.loc[clean_splitting_df["event"] == 'break', "time_delta"] = 0
    
    splits = []
    delta = 0
    for time_diff, index, event in zip(clean_splitting_df['time_delta'], clean_splitting_df['index'], clean_splitting_df['event']):
        if event == 'break':
            splits.append(index)
            delta = 0
        elif time_diff > min_event_time or delta > min_event_time:
            splits.append(index)
            delta = 0
        else:
            delta += time_diff

    splits[0] = 0
    splits.append(events_df['index'].iloc[-1])

    idx_lst = [] 
    for i in range(len(splits)):
        if splits[i] == 0:
            idx_lst.append([splits[i], splits[i+1]])
        else:
            idx_lst.append([splits[i], splits[i+1]])
        if splits[i+1] == splits[-1]:
            break
            
    idx_lst[-1][-1] = idx_lst[-1][-1]+1

    df_lst = []
    for i in idx_lst:
        df = events_df.iloc[i[0]:i[1],:]
        df_lst.append(df)

    return df_lst

def get_stats_from_network(G, stats_all=False, plotting=False, axes=None, curr_idx = None):
    """
    Takes: graph (from fx.ReturnNXPassNetwork), stats_all (optional) = bool, plotting (optional) = bool
    l_r (last row) (optional) = int, axes = optional
    Returns: Pandas DataFrame of network statistics of the team
    """
    pathlengths = []

    for v in G.nodes():
        spl = dict(nx.single_source_shortest_path_length(G, v))
        for p in spl:
            pathlengths.append(spl[p])
    dist = {}
    for p in pathlengths:
        if p in dist:
            dist[p] += 1
        else:
            dist[p] = 1
    verts = dist.keys()
    clustering=[]
    names=[]
    for i in G.nodes:
        names.append('Clustering of '+i)
        clustering.append(nx.clustering(G,i))
    if not stats_all:
        data={'Properties':(['Strongly connected','Weakly connected','Average shortest path length',
        'Radius','Diameter','Average Eccentricity','Center',
        'Average Eigenvector centrality','Periphery','Density',
        'Average Clustering']),
        'Values':([nx.is_strongly_connected(G),nx.is_weakly_connected(G),nx.average_shortest_path_length(G),
        nx.radius(G), nx.diameter(G),statistics.mean(list(nx.eccentricity(G).values())),nx.center(G)[0],
        statistics.mean(list(nx.eigenvector_centrality(G).values())),nx.periphery(G),nx.density(G),
        statistics.mean(list(nx.clustering(G).values()))])}
    else:
        data={'Strongly connected': [nx.is_strongly_connected(G)], 'Weakly connected': [nx.is_weakly_connected(G)], 
            'Average shortest path length': [nx.average_shortest_path_length(G)], 'Radius': [nx.radius(G)],
            'Diameter' : [nx.diameter(G)], 'Average Eccentricity' : [statistics.mean(list(nx.eccentricity(G).values()))],
            'Center' : nx.center(G)[0],
            'Average Eigenvector centrality': [statistics.mean(list(nx.eigenvector_centrality(G).values()))],
            'Periphery' : nx.periphery(G),'Density' : [nx.density(G)],
            'Average Clustering': [statistics.mean(list(nx.clustering(G).values()))]} # 'Core' : nx.core_number(G)
        clustrs = list(zip(names,clustering))
        df = pd.DataFrame(data)
        for pair in clustrs:
            df[pair[0]] = [pair[1]]  
    if plotting:
        dist = pd.DataFrame({'path_length':list(dist.keys()), 'count':list(dist.values())})
        if curr_idx:
            sns.barplot(data=dist, x='path_length', y='count', ax=axes[0 + curr_idx*2]).set(title='Counts of shortest path lengths')
        else:
            sns.barplot(data=dist, x='path_length', y='count', ax=axes[0]).set(title='Counts of shortest path lengths')
        kcores = defaultdict(list)
        for n, k in nx.core_number(G).items():
            kcores[k].append(n) #append node to appropriate core
        max_core = int(max(kcores.keys()))
        pos = nx.layout.shell_layout(G, list(kcores.values()))
        for node in G.nodes():
            cmap = plt.get_cmap('plasma')
            if curr_idx:
                nx.draw_networkx_nodes(G, pos, nodelist=[node],
                node_color=np.array([cmap(nx.core_number(G)[node]/max_core)]), ax=axes[1 + curr_idx*2]) #plt magic <3,  numpy array is there to get rid of a warning
                nx.draw_networkx_labels(G,pos,labels={node:nx.core_number(G)[node] for node in G.nodes()}, ax=axes[1 + curr_idx*2])
            else:
                nx.draw_networkx_nodes(G, pos, nodelist=[node],
                node_color=np.array([cmap(nx.core_number(G)[node]/max_core)]), ax=axes[1]) #forgive the if's :(
                nx.draw_networkx_labels(G,pos,labels={node:nx.core_number(G)[node] for node in G.nodes()}, ax=axes[1])  
        maxWeight = max([edge[2]['weight'] for edge in list(G.edges(data=True))])   
        if curr_idx:
            for edge in G.edges(data='weight'):
                nx.draw_networkx_edges(G, pos, edgelist=[edge],width=(edge[2]/maxWeight+1)**2,
                                        alpha=edge[2]/maxWeight, arrows="FancyArrowPatch", ax=axes[1 + curr_idx*2])
        else:
            for edge in G.edges(data='weight'):
                nx.draw_networkx_edges(G, pos, edgelist=[edge],width=(edge[2]/maxWeight+1)**2,
                                        alpha=edge[2]/maxWeight, arrows="FancyArrowPatch", ax=axes[1])
        
    if not ('df' in locals()): df = pd.DataFrame(data)

    return df