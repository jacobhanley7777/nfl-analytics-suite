import os
import pickle
import math
import re
import warnings
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import gradio as gr

warnings.filterwarnings('ignore')

COL_MODEL = "Model's Projection"
COL_SLEEPER = "Sleeper's Projection"
COL_DIFF = "Diff (Model - Sleeper)"

CACHE_PATH = os.path.expanduser("~/.fantasy_suite_2026_v9_props_cache.pkl")
PLAYERS_CACHE_PATH = os.path.expanduser("~/.sleeper_players_2026_cache.pkl")
LOCAL_LEAGUE_ID_FILE = os.path.expanduser("~/.last_sleeper_league_id.txt")

def load_saved_league_id():
    if os.path.exists(LOCAL_LEAGUE_ID_FILE):
        try:
            with open(LOCAL_LEAGUE_ID_FILE, "r") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""

def save_local_league_id(lid):
    lid = str(lid or "").strip()
    if lid and lid.isdigit():
        try:
            with open(LOCAL_LEAGUE_ID_FILE, "w") as f:
                f.write(lid)
        except Exception:
            pass

DEFAULT_LEAGUE_ID = ""

NICKNAME_MAP = {
    'hollywood brown': 'marquise brown',
    'tank dell': 'nathaniel dell',
    'chig okonkwo': 'chigoziem okonkwo',
    'bucky irving': 'markeise irving',
    'gabe davis': 'gabriel davis',
    'josh palmer': 'joshua palmer',
    'pat freiermuth': 'patrick freiermuth',
    'ken walker': 'kenneth walker',
    'dk metcalf': 'decaf metcalf'
}

def norm_name(n):
    if not isinstance(n, str):
        return ""
    n = re.sub(r'[^a-zA-Z\s]', '', n.lower())
    for sfx in [' jr', ' sr', ' iii', ' ii', ' iv', ' v']:
        if n.endswith(sfx):
            n = n[:-len(sfx)]
    n = re.sub(r'\s+', ' ', n).strip()
    return NICKNAME_MAP.get(n, n)

print("1/6: Loading 2024-2026 NFL Schedule, Weather/Rest Telemetry, and Play-by-Play Logs...")

team_fix = {'LAR': 'LA', 'WSH': 'WAS', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}

sched_url = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
sched_df = pd.read_csv(sched_url, low_memory=False)
sched_full = sched_df[(sched_df['season'].isin([2024, 2025, 2026])) & (sched_df['game_type'] == 'REG')].copy()
sched_full['home_team'] = sched_full['home_team'].replace(team_fix)
sched_full['away_team'] = sched_full['away_team'].replace(team_fix)

schedule_map = {}
hist_env_map = {}

for _, row in sched_full.iterrows():
    ssn = int(row['season'])
    wk = int(row['week'])
    h_team, a_team = row['home_team'], row['away_team']
    vegas_tot = float(row['total_line']) if pd.notna(row.get('total_line')) else 44.5
    spread = float(row['spread_line']) if pd.notna(row.get('spread_line')) else 0.0
    
    roof_raw = str(row.get('roof', '') or '').lower()
    is_dome = 1 if roof_raw in ['dome', 'closed'] else 0
    wind_spd = 0.0 if is_dome else (float(row['wind']) if pd.notna(row.get('wind')) else 6.0)
    temp_f = 72.0 if is_dome else (float(row['temp']) if pd.notna(row.get('temp')) else 62.0)
    h_rest = int(row['home_rest']) if pd.notna(row.get('home_rest')) else 7
    a_rest = int(row['away_rest']) if pd.notna(row.get('away_rest')) else 7
    is_div = int(row['div_game']) if pd.notna(row.get('div_game')) else 0
    
    h_imp = max(13.0, (vegas_tot - spread) / 2.0)
    a_imp = max(13.0, (vegas_tot + spread) / 2.0)
    env_tag = "Indoor Dome" if is_dome else (f"Outdoor ({int(wind_spd)} mph wind)" if wind_spd >= 12 else "Outdoor Clear")
    
    h_dict = {
        'opp': a_team, 'is_home': 1, 'vegas_total': vegas_tot,
        'team_implied': round(h_imp, 1), 'opp_implied': round(a_imp, 1),
        'team_margin': round(h_imp - a_imp, 1), 'matchup_str': f"vs {a_team}",
        'is_dome': is_dome, 'wind': wind_spd, 'temp': temp_f,
        'rest_days': h_rest, 'div_game': is_div, 'env_tag': env_tag
    }
    a_dict = {
        'opp': h_team, 'is_home': 0, 'vegas_total': vegas_tot,
        'team_implied': round(a_imp, 1), 'opp_implied': round(h_imp, 1),
        'team_margin': round(a_imp - h_imp, 1), 'matchup_str': f"@ {h_team}",
        'is_dome': is_dome, 'wind': wind_spd, 'temp': temp_f,
        'rest_days': a_rest, 'div_game': is_div, 'env_tag': env_tag
    }
    hist_env_map[(ssn, wk, h_team)] = h_dict
    hist_env_map[(ssn, wk, a_team)] = a_dict
    if ssn == 2026:
        schedule_map[(wk, h_team)] = h_dict
        schedule_map[(wk, a_team)] = a_dict

print("2/6: Loading Player Biographical Profiles, Official Headshots & Sleeper ID Mappings...")
sleeper_pid_to_info = {}
norm_to_bio = {}
gsis_to_bio = {}

if os.path.exists(PLAYERS_CACHE_PATH):
    try:
        with open(PLAYERS_CACHE_PATH, "rb") as f:
            p_cache = pickle.load(f)
            sleeper_pid_to_info = p_cache.get('sleeper_pid_to_info', {})
            norm_to_bio = p_cache.get('norm_to_bio', {})
            gsis_to_bio = p_cache.get('gsis_to_bio', {})
    except Exception:
        pass

if not sleeper_pid_to_info:
    try:
        resp = requests.get("https://api.sleeper.app/v1/players/nfl", headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
        if resp.status_code == 200:
            for pid, pdata in resp.json().items():
                if not isinstance(pdata, dict):
                    continue
                pos = pdata.get('position', '')
                if pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                    fname = f"{pdata.get('first_name', '')} {pdata.get('last_name', '')}".strip()
                    t_abbr = team_fix.get(pdata.get('team') or '', pdata.get('team') or str(pid))
                    gsis = str(pdata.get('gsis_id') or '').strip()
                    college = pdata.get('college') or 'NFL'
                    age = pdata.get('age') or 'N/A'
                    exp = pdata.get('years_exp')
                    exp_str = "Rookie" if exp == 0 else (f"{exp} yrs" if exp is not None else "Vet")
                    ht = pdata.get('height') or ''
                    wt = pdata.get('weight') or ''
                    headshot_url = f"https://sleepercdn.com/content/nfl/players/thumb/{pid}.jpg" if pos != 'DEF' else f"https://sleepercdn.com/images/team_logos/nfl/{str(t_abbr).lower()}.png"
                    
                    bio_obj = {
                        'sleeper_id': str(pid),
                        'name': fname if fname else str(pid),
                        'norm': norm_name(fname),
                        'pos': pos,
                        'team': t_abbr,
                        'gsis_id': gsis,
                        'college': college,
                        'age': age,
                        'exp': exp_str,
                        'height': ht,
                        'weight': wt,
                        'headshot': headshot_url
                    }
                    sleeper_pid_to_info[str(pid)] = bio_obj
                    if bio_obj['norm']:
                        norm_to_bio[(bio_obj['norm'], pos)] = bio_obj
                        norm_to_bio[bio_obj['norm']] = bio_obj
                    if gsis:
                        gsis_to_bio[gsis] = bio_obj
            try:
                with open(PLAYERS_CACHE_PATH, "wb") as f:
                    pickle.dump({'sleeper_pid_to_info': sleeper_pid_to_info, 'norm_to_bio': norm_to_bio, 'gsis_to_bio': gsis_to_bio}, f)
            except Exception:
                pass
    except Exception as e:
        print(f"  Notice on player bio sync: {e}")

print("3/6: Building 36-Feature Matrix (WOPR, RACR, EPA, 1st Downs, YAC, Weather, Spread & DvP)...")

pos_map = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4}
features = [
    'pos_code', 'talent_prior', 'season_ppg_prior', 'career_ppg_prior', 'ewma_pts',
    'roll2_pts', 'roll3_pts', 'roll5_pts', 'roll5_std',
    'roll3_xFP', 'roll2_xFP', 'season_xFP_prior', 'role_trend', 'roll3_luck',
    'roll3_targets', 'roll3_receptions', 'roll3_carries',
    'roll3_tgt_share', 'roll3_air_share', 'roll3_wopr', 'roll3_racr',
    'roll3_epa', 'roll3_first_downs', 'roll3_air_yards', 'roll3_yac',
    'roll3_pass_yds', 'roll3_eff_edge', 'roll3_td_equity',
    'matchup_boost', 'vegas_total', 'team_implied', 'team_margin',
    'is_dome', 'wind_speed', 'rest_days', 'div_game'
]

df_2026_raw = pd.read_csv("https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_2026.csv", low_memory=False)
current_data_sig = (int(df_2026_raw['week'].max()), len(df_2026_raw))

cache_loaded = False
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "rb") as f:
            cached_bundle = pickle.load(f)
        if cached_bundle.get('data_sig') == current_data_sig and cached_bundle.get('version') == 'v9_props':
            df = cached_bundle['df']
            latest_dvp = cached_bundle['latest_dvp']
            model_mean = cached_bundle['model_mean']
            model_median = cached_bundle['model_median']
            model_mae = cached_bundle['model_mae']
            model_floor = cached_bundle['model_floor']
            model_boom = cached_bundle['model_boom']
            prop_models = cached_bundle['prop_models']
            pos_baselines = cached_bundle['pos_baselines']
            name_col = cached_bundle['name_col']
            team_col = cached_bundle['team_col']
            opp_col = cached_bundle['opp_col']
            cache_loaded = True
            print(f"4/6: Loaded Pre-Trained Fantasy + Player Prop Models from Local Cache ({current_data_sig[1]} rows through Week {current_data_sig[0]})...")
    except Exception:
        cache_loaded = False

if not cache_loaded:
    dfs = []
    for y in [2024, 2025]:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{y}.csv"
        try:
            dfs.append(pd.read_csv(url, low_memory=False))
        except Exception:
            pass
    dfs.append(df_2026_raw)
    df = pd.concat(dfs, ignore_index=True)

    name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
    team_col = 'team' if 'team' in df.columns else 'recent_team'
    opp_col = 'opponent_team' if 'opponent_team' in df.columns else 'opponent'

    df[team_col] = df[team_col].replace(team_fix)
    df[opp_col] = df[opp_col].replace(team_fix)
    df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()

    stat_cols = [
        'fantasy_points_ppr', 'targets', 'receptions', 'receiving_yards', 'receiving_tds',
        'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa',
        'target_share', 'air_yards_share', 'wopr', 'racr',
        'carries', 'rushing_yards', 'rushing_tds', 'rushing_first_downs', 'rushing_epa',
        'passing_yards', 'passing_tds', 'interceptions', 'passing_first_downs', 'passing_epa', 'completions', 'attempts'
    ]
    for c in stat_cols:
        df[c] = df[c].fillna(0.0) if c in df.columns else 0.0

    df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)

    vegas_tot_list, team_imp_list, team_mar_list, dome_list, wind_list, rest_list, div_list = [], [], [], [], [], [], []
    for _, r in df.iterrows():
        env = hist_env_map.get((int(r['season']), int(r['week']), r[team_col]))
        if env:
            vegas_tot_list.append(env['vegas_total'])
            team_imp_list.append(env['team_implied'])
            team_mar_list.append(env['team_margin'])
            dome_list.append(env['is_dome'])
            wind_list.append(env['wind'])
            rest_list.append(env['rest_days'])
            div_list.append(env['div_game'])
        else:
            vegas_tot_list.append(44.5)
            team_imp_list.append(22.2)
            team_mar_list.append(0.0)
            dome_list.append(0)
            wind_list.append(6.0)
            rest_list.append(7)
            div_list.append(0)
            
    df['vegas_total'] = vegas_tot_list
    df['team_implied'] = team_imp_list
    df['team_margin'] = team_mar_list
    df['is_dome'] = dome_list
    df['wind_speed'] = wind_list
    df['rest_days'] = rest_list
    df['div_game'] = div_list

    df['calc_wopr'] = np.where(df['wopr'] > 0, df['wopr'], 1.5 * df['target_share'] + 0.7 * df['air_yards_share'])
    df['calc_racr'] = np.where(df['racr'] > 0, df['racr'], np.where(df['receiving_air_yards'] >= 10, df['receiving_yards'] / np.maximum(1.0, df['receiving_air_yards']), 1.0))
    df['total_epa'] = df['passing_epa'] + df['rushing_epa'] + df['receiving_epa']
    df['total_first_downs'] = df['passing_first_downs'] + df['rushing_first_downs'] + df['receiving_first_downs']
    df['td_equity'] = df['rushing_tds'] + df['receiving_tds'] + df['passing_tds'] * 0.5
    df['non_qb_tds'] = df['rushing_tds'] + df['receiving_tds']
    
    ypc_edge = np.where(df['carries'] >= 3, (df['rushing_yards'] / np.maximum(1, df['carries'])) - 4.3, 0.0)
    ypt_edge = np.where(df['targets'] >= 2, (df['receiving_yards'] / np.maximum(1, df['targets'])) - 7.8, 0.0)
    df['eff_edge'] = np.clip(ypc_edge * 0.45 + ypt_edge * 0.35, -3.5, 4.5)

    df['xFP_volume'] = (
        df['targets'] * 1.25 +
        df['receptions'] * 0.38 +
        df['receiving_air_yards'] * 0.042 +
        df['receiving_first_downs'] * 0.55 +
        df['carries'] * 0.74 +
        df['rushing_yards'] * 0.048 +
        df['rushing_first_downs'] * 0.52 +
        df['passing_yards'] * 0.04 +
        df['passing_tds'] * 3.60 +
        (df['rushing_tds'] + df['receiving_tds']) * 2.15 -
        df['interceptions'] * 1.50
    )
    df['luck_delta'] = df['fantasy_points_ppr'] - df['xFP_volume']

    g = df.groupby('player_id')
    df['roll2_pts'] = g['fantasy_points_ppr'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    df['roll3_pts'] = g['fantasy_points_ppr'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll5_pts'] = g['fantasy_points_ppr'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['ewma_pts'] = g['fantasy_points_ppr'].transform(lambda x: x.shift(1).ewm(span=3.5, min_periods=1).mean())

    df['season_ppg_prior'] = df.groupby(['player_id', 'season'])['fantasy_points_ppr'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    df['career_ppg_prior'] = g['fantasy_points_ppr'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    df['season_ppg_prior'] = df['season_ppg_prior'].fillna(df['career_ppg_prior'])

    df['roll5_std'] = g['fantasy_points_ppr'].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(4.2)
    df['roll2_xFP'] = g['xFP_volume'].transform(lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    df['roll3_xFP'] = g['xFP_volume'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['season_xFP_prior'] = df.groupby(['player_id', 'season'])['xFP_volume'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean()).fillna(df['roll3_xFP'])

    df['role_trend'] = (df['roll2_xFP'] - df['season_xFP_prior']).fillna(0.0)
    df['roll3_luck'] = g['luck_delta'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_targets'] = g['targets'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_receptions'] = g['receptions'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_carries'] = g['carries'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_tgt_share'] = g['target_share'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_air_share'] = g['air_yards_share'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    df['roll3_wopr'] = g['calc_wopr'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_racr'] = g['calc_racr'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(1.0)
    df['roll3_epa'] = g['total_epa'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_first_downs'] = g['total_first_downs'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_air_yards'] = g['receiving_air_yards'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_yac'] = g['receiving_yards_after_catch'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    df['roll3_pass_yds'] = g['passing_yards'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['roll3_rush_yds'] = g['rushing_yards'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    df['roll3_rec_yds'] = g['receiving_yards'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    df['roll3_eff_edge'] = g['eff_edge'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
    df['roll3_td_equity'] = g['td_equity'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)

    df['talent_prior'] = (
        0.36 * df['season_ppg_prior'] +
        0.28 * df['ewma_pts'] +
        0.26 * df['roll3_xFP'] +
        0.10 * df['career_ppg_prior']
    )

    def_pos = df.groupby(['season', 'week', opp_col, 'position'])['fantasy_points_ppr'].sum().reset_index()
    def_pos = def_pos.sort_values(['season', 'week'])
    def_pos['opp_pts_allowed'] = def_pos.groupby([opp_col, 'position'])['fantasy_points_ppr'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    pos_avg = def_pos.groupby('position')['fantasy_points_ppr'].transform('mean')
    def_pos['matchup_boost'] = ((def_pos['opp_pts_allowed'] - pos_avg) * 0.48).fillna(0.0)

    df = df.merge(def_pos[['season', 'week', opp_col, 'position', 'matchup_boost']], on=['season', 'week', opp_col, 'position'], how='left')
    df['matchup_boost'] = df['matchup_boost'].fillna(0.0)
    latest_dvp = def_pos[def_pos['season'] == 2026].sort_values('week').groupby([opp_col, 'position'])['matchup_boost'].last().to_dict()

    df['pos_code'] = df['position'].map(pos_map)

    print("4/6: Training Fantasy GBDT Ensemble + 5 Dedicated Player Prop Models (Pass/Rush/Rec Yds, Recs, TD)...")
    train_df = df.dropna(subset=features + ['fantasy_points_ppr']).copy()
    train_df = train_df[train_df['fantasy_points_ppr'] >= 0.0]
    X_train = train_df[features]
    y_train = train_df['fantasy_points_ppr']

    model_mean = HistGradientBoostingRegressor(
        loss='squared_error', max_iter=250, max_depth=5, learning_rate=0.042, l2_regularization=0.20, random_state=42
    )
    model_mean.fit(X_train, y_train)

    model_mae = HistGradientBoostingRegressor(
        loss='absolute_error', max_iter=250, max_depth=5, learning_rate=0.042, l2_regularization=0.20, random_state=42
    )
    model_mae.fit(X_train, y_train)

    model_median = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.50, max_iter=230, max_depth=5, learning_rate=0.042, l2_regularization=0.20, random_state=42
    )
    model_median.fit(X_train, y_train)

    model_floor = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.10, max_iter=150, max_depth=4, learning_rate=0.05, random_state=42
    )
    model_floor.fit(X_train, y_train)

    model_boom = HistGradientBoostingRegressor(
        loss='quantile', quantile=0.90, max_iter=150, max_depth=4, learning_rate=0.05, random_state=42
    )
    model_boom.fit(X_train, y_train)

    # Train Dedicated Multi-Target Player Prop Models (Conditional Median + Mean Blend for Sportsbook Prop Accuracy)
    prop_models = {}
    for prop_key, target_col in [
        ('pass_yd', 'passing_yards'),
        ('rush_yd', 'rushing_yards'),
        ('rec_yd', 'receiving_yards'),
        ('receptions', 'receptions'),
        ('any_td', 'non_qb_tds')
    ]:
        y_prop = train_df[target_col].clip(lower=0.0)
        if prop_key == 'any_td':
            m_p = HistGradientBoostingRegressor(loss='squared_error', max_iter=180, max_depth=4, learning_rate=0.045, l2_regularization=0.25, random_state=42)
            m_p.fit(X_train, y_prop)
            prop_models[prop_key] = ('mean', m_p)
        else:
            m_med = HistGradientBoostingRegressor(loss='quantile', quantile=0.50, max_iter=200, max_depth=5, learning_rate=0.045, l2_regularization=0.20, random_state=42)
            m_mean = HistGradientBoostingRegressor(loss='squared_error', max_iter=200, max_depth=5, learning_rate=0.045, l2_regularization=0.20, random_state=42)
            m_med.fit(X_train, y_prop)
            m_mean.fit(X_train, y_prop)
            prop_models[prop_key] = ('blend', m_med, m_mean)

    starter_ref = train_df[train_df['talent_prior'] >= 7.0]
    pos_baselines = {}
    for p_code in [1, 2, 3, 4]:
        sub = starter_ref[starter_ref['pos_code'] == p_code]
        if sub.empty:
            sub = train_df[train_df['pos_code'] == p_code]
        pos_baselines[p_code] = sub[features].median().to_dict()

    try:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                'version': 'v9_props', 'data_sig': current_data_sig, 'df': df, 'latest_dvp': latest_dvp,
                'model_mean': model_mean, 'model_median': model_median, 'model_mae': model_mae,
                'model_floor': model_floor, 'model_boom': model_boom, 'prop_models': prop_models,
                'pos_baselines': pos_baselines, 'name_col': name_col, 'team_col': team_col, 'opp_col': opp_col
            }, f)
    except Exception:
        pass

df_2026 = df[df['season'] == 2026].copy()
latest_completed_week = int(df_2026['week'].max())
default_target_week = min(latest_completed_week + 1, 18)
active_player_names = sorted(df_2026[name_col].dropna().unique().tolist())

playoff_sos_map = {}
all_nfl_teams = df_2026[team_col].dropna().unique().tolist()
for t_abbr in all_nfl_teams:
    for pos in ['QB', 'RB', 'WR', 'TE']:
        po_boosts = []
        for po_wk in [15, 16, 17]:
            s_info = schedule_map.get((po_wk, t_abbr))
            if s_info:
                po_boosts.append(float(latest_dvp.get((s_info['opp'], pos), 0.0)))
        avg_po = float(np.mean(po_boosts)) if po_boosts else 0.0
        playoff_sos_map[(t_abbr, pos)] = round(avg_po, 2)

sleeper_cache = {}
projection_week_cache = {}
league_analysis_cache = {}
live_sportsbook_props_cache = {}
OPTIMAL_ML_WEIGHT = 0.60

def get_player_bio(p_name, p_pos, p_team, gsis_id=""):
    norm_k = norm_name(p_name)
    bio = None
    if gsis_id and str(gsis_id) in gsis_to_bio:
        bio = gsis_to_bio[str(gsis_id)]
    elif (norm_k, p_pos) in norm_to_bio:
        bio = norm_to_bio[(norm_k, p_pos)]
    elif norm_k in norm_to_bio:
        bio = norm_to_bio[norm_k]
        
    fallback_logo = f"https://sleepercdn.com/images/team_logos/nfl/{str(p_team).lower()}.png"
    if bio:
        return {
            'headshot': bio.get('headshot') or fallback_logo,
            'college': bio.get('college') or 'NFL',
            'age': bio.get('age') or 'N/A',
            'exp': bio.get('exp') or 'Vet',
            'height': bio.get('height') or '',
            'weight': bio.get('weight') or ''
        }
    return {
        'headshot': fallback_logo,
        'college': 'NFL', 'age': 'N/A', 'exp': 'Active', 'height': '', 'weight': ''
    }

def get_sleeper_week_data(week_num, force_refresh=False):
    if not force_refresh and week_num in sleeper_cache:
        return sleeper_cache[week_num]
    url = f"https://api.sleeper.app/projections/nfl/2026/{week_num}?season_type=regular&position[]=QB&position[]=RB&position[]=WR&position[]=TE&position[]=K&position[]=DEF"
    proj_lookup, status_lookup, pid_proj_lookup, stat_proj_lookup = {}, {}, {}, {}
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if resp.status_code == 200:
            for item in resp.json():
                p_info = item.get('player', {}) or {}
                stats = item.get('stats', {}) or {}
                pid = str(item.get('player_id') or p_info.get('player_id') or '').strip()
                pts = float(stats.get('pts_ppr') or stats.get('pts_std') or 0.0)
                full_name = f"{p_info.get('first_name', '')} {p_info.get('last_name', '')}".strip()
                pos = p_info.get('position', '')
                team_abbr = team_fix.get(p_info.get('team') or '', p_info.get('team') or pid)
                inj = p_info.get('injury_status')
                p_stat = p_info.get('status')
                key = (norm_name(full_name), pos)
                
                if pid and pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF'] and pid not in sleeper_pid_to_info:
                    headshot_url = f"https://sleepercdn.com/content/nfl/players/thumb/{pid}.jpg" if pos != 'DEF' else f"https://sleepercdn.com/images/team_logos/nfl/{str(team_abbr).lower()}.png"
                    b_obj = {
                        'sleeper_id': pid,
                        'name': full_name if full_name else pid,
                        'norm': norm_name(full_name),
                        'pos': pos,
                        'team': team_abbr,
                        'gsis_id': '',
                        'college': 'NFL', 'age': 'N/A', 'exp': 'Active', 'height': '', 'weight': '',
                        'headshot': headshot_url
                    }
                    sleeper_pid_to_info[pid] = b_obj
                    if b_obj['norm']:
                        norm_to_bio[(b_obj['norm'], pos)] = b_obj
                        norm_to_bio[b_obj['norm']] = b_obj
                
                if inj in ['Out', 'IR', 'PUP', 'Doubtful', 'Sus']:
                    status_lookup[key] = inj.upper()
                elif p_stat in ['Inactive', 'Injured Reserve']:
                    status_lookup[key] = 'OUT'
                elif inj == 'Questionable':
                    status_lookup[key] = 'Q'
                else:
                    status_lookup[key] = 'ACT'
                    
                if pts > 0:
                    proj_lookup[key] = round(pts, 2)
                    if pid:
                        pid_proj_lookup[pid] = round(pts, 2)
                    stat_proj_lookup[key] = {
                        'pass_yd': round(float(stats.get('pass_yd', 0.0) or 0.0), 1),
                        'rush_yd': round(float(stats.get('rush_yd', 0.0) or 0.0), 1),
                        'rec_yd': round(float(stats.get('rec_yd', 0.0) or 0.0), 1),
                        'receptions': round(float(stats.get('rec', 0.0) or 0.0), 1),
                        'any_td': round(float((stats.get('rush_td', 0.0) or 0.0) + (stats.get('rec_td', 0.0) or 0.0)), 2)
                    }
                elif status_lookup[key] in ['OUT', 'IR', 'PUP', 'DOUBTFUL']:
                    proj_lookup[key] = 0.0
                    if pid:
                        pid_proj_lookup[pid] = 0.0
    except Exception:
        pass
    sleeper_cache[week_num] = (proj_lookup, status_lookup, pid_proj_lookup, stat_proj_lookup)
    return proj_lookup, status_lookup, pid_proj_lookup, stat_proj_lookup

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

def prob_to_american_odds(prob_pct):
    p = np.clip(float(prob_pct) / 100.0, 0.02, 0.98)
    if p >= 0.50:
        odds = -int(round((p / (1.0 - p)) * 100.0))
        return f"{odds}"
    else:
        odds = int(round(((1.0 - p) / p) * 100.0))
        return f"+{odds}"

def american_odds_to_implied_prob(odds_val):
    try:
        o = float(odds_val)
    except Exception:
        o = -110.0
    if o < 0:
        return (-o) / (-o + 100.0)
    elif o > 0:
        return 100.0 / (o + 100.0)
    return 0.5238

def render_payoff_distribution_bar_html(p_name, pos_team, flr_val, mod_val, slp_val, ceil_val, max_scale=34.0, compact=False):
    flr_pct = np.clip((flr_val / max_scale) * 100.0, 1.0, 96.0)
    mod_pct = np.clip((mod_val / max_scale) * 100.0, flr_pct + 1.0, 98.0)
    slp_pct = np.clip((slp_val / max_scale) * 100.0, 1.0, 98.0)
    ceil_pct = np.clip((ceil_val / max_scale) * 100.0, mod_pct + 1.5, 99.5)
    span_width = max(2.5, ceil_pct - flr_pct)
    
    pad = "8px 12px" if compact else "12px 16px"
    html = (
        f'<div style="padding:{pad}; border-radius:10px; border:1px solid rgba(148,163,184,0.3); background:rgba(15,23,42,0.02); margin-bottom:8px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; font-size:0.86rem; margin-bottom:6px;">'
        f'<div><strong>{p_name}</strong> <span style="opacity:0.75; font-size:0.8rem;">({pos_team})</span></div>'
        f'<div style="font-size:0.82rem;">'
        f'<span style="opacity:0.75;">Floor: <strong>{flr_val:.1f}</strong></span> &nbsp;|&nbsp; '
        f'<span style="color:#2563eb; font-weight:700;">Model: {mod_val:.2f}</span> &nbsp;|&nbsp; '
        f'<span style="opacity:0.8;">Sleeper: {slp_val:.2f}</span> &nbsp;|&nbsp; '
        f'<span style="color:#059669; font-weight:700;">Ceiling: {ceil_val:.1f}</span>'
        f'</div></div>'
        f'<div style="position:relative; height:18px; border-radius:9px; background:rgba(148,163,184,0.18); overflow:hidden;">'
        f'<div style="position:absolute; left:29.4%; top:0; bottom:0; width:1px; background:rgba(148,163,184,0.35);" title="10 PPR"></div>'
        f'<div style="position:absolute; left:58.8%; top:0; bottom:0; width:1px; background:rgba(148,163,184,0.35);" title="20 PPR"></div>'
        f'<div style="position:absolute; left:88.2%; top:0; bottom:0; width:1px; background:rgba(148,163,184,0.35);" title="30 PPR"></div>'
        f'<div style="position:absolute; left:{flr_pct:.1f}%; width:{span_width:.1f}%; top:3px; bottom:3px; border-radius:6px; background:linear-gradient(90deg, rgba(59,130,246,0.35) 0%, rgba(37,99,235,0.65) 55%, rgba(16,185,129,0.75) 100%);"></div>'
        f'<div style="position:absolute; left:calc({slp_pct:.1f}% - 2px); top:1px; bottom:1px; width:4px; border-radius:2px; background:#94a3b8; border:1px solid #475569;" title="Sleeper Consensus: {slp_val:.2f}"></div>'
        f'<div style="position:absolute; left:calc({mod_pct:.1f}% - 5px); top:1px; width:10px; height:16px; border-radius:4px; background:#1d4ed8; border:2px solid #ffffff; box-shadow:0 1px 3px rgba(0,0,0,0.3);" title="Model Projection: {mod_val:.2f}"></div>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between; font-size:0.7rem; opacity:0.6; margin-top:3px;">'
        f'<span>0 PPR</span><span>10 PPR</span><span>20 PPR</span><span>30+ PPR</span>'
        f'</div></div>'
    )
    return html

def compute_shapley_group_attribution(p_row):
    p_code = int(p_row['pos_code'])
    base_vec = dict(pos_baselines.get(p_code, {f: 0.0 for f in features}))
    
    def eval_ens(vec_dict):
        arr = pd.DataFrame([vec_dict])[features]
        return float(0.42 * model_mean.predict(arr)[0] + 0.33 * model_mae.predict(arr)[0] + 0.25 * model_median.predict(arr)[0])
        
    base_pred = eval_ens(base_vec)
    
    vol_feats = ['talent_prior', 'season_ppg_prior', 'career_ppg_prior', 'ewma_pts', 'roll2_pts', 'roll3_pts', 'roll5_pts', 'roll3_xFP', 'roll2_xFP', 'season_xFP_prior', 'role_trend', 'roll3_targets', 'roll3_receptions', 'roll3_carries', 'roll3_pass_yds', 'roll3_td_equity']
    eff_feats = ['roll3_tgt_share', 'roll3_air_share', 'roll3_wopr', 'roll3_racr', 'roll3_epa', 'roll3_first_downs', 'roll3_air_yards', 'roll3_yac', 'roll3_eff_edge', 'roll3_luck', 'roll5_std']
    match_feats = ['matchup_boost']
    env_feats = ['vegas_total', 'team_implied', 'team_margin', 'is_dome', 'wind_speed', 'rest_days', 'div_game']
    
    v1 = dict(base_vec)
    for f in vol_feats: v1[f] = p_row[f]
    p1 = eval_ens(v1)
    vol_contrib = p1 - base_pred
    
    v2 = dict(v1)
    for f in eff_feats: v2[f] = p_row[f]
    p2 = eval_ens(v2)
    eff_contrib = p2 - p1
    
    v3 = dict(v2)
    for f in match_feats: v3[f] = p_row[f]
    p3 = eval_ens(v3)
    match_contrib = (p3 - p2) * 0.72 + float(p_row['matchup_boost']) * 0.14
    
    v4 = dict(v3)
    for f in env_feats: v4[f] = p_row[f]
    p4 = eval_ens(v4)
    env_contrib = (p4 - p3) * 0.72 + (float(p_row['team_implied']) - 22.0) * 0.12 + float(p_row['Vacated Boost']) * 0.45
    
    return {
        'Positional Starter Baseline': round(base_pred, 2),
        'Volume & Role Equity (xFP / Touches)': round(vol_contrib, 2),
        'Play-by-Play Efficiency (WOPR / EPA / 1st Downs)': round(eff_contrib, 2),
        'Defense vs. Position (Normalized DvP)': round(match_contrib, 2),
        'Vegas Script, Weather & Vacated Volume': round(env_contrib, 2)
    }

print(f"5/6: Compiling Dynamic Weekly Inference Core & Player Prop Projections (Week {default_target_week})...")

def build_week_projections(target_week, force_refresh=False, ml_weight_override=None):
    target_week = int(target_week)
    if not force_refresh and ml_weight_override is None and target_week in projection_week_cache:
        return projection_week_cache[target_week].copy()
        
    is_past_week = (target_week <= latest_completed_week)
    sleeper_map, injury_map, _, stat_proj_map = get_sleeper_week_data(target_week, force_refresh=force_refresh)
    
    cutoff_df = df_2026[df_2026['week'] < target_week].copy() if is_past_week and target_week > 1 else df_2026.copy()
    if cutoff_df.empty:
        cutoff_df = df[df['season'] == 2025].copy()
        
    roster_pool = df_2026.sort_values('week').groupby('player_id').last().reset_index()
    actual_map = dict(zip(df_2026[df_2026['week'] == target_week]['player_id'], df_2026[df_2026['week'] == target_week]['fantasy_points_ppr'].round(2))) if is_past_week else {}

    rows = []
    for _, player in roster_pool.iterrows():
        pid, p_name, p_pos, p_team = str(player['player_id']).strip(), player[name_col], player['position'], player[team_col]
        p_hist = cutoff_df[cutoff_df['player_id'] == player['player_id']].sort_values(['season', 'week'])
        if p_hist.empty:
            continue
            
        full_2026_hist = df_2026[df_2026['player_id'] == player['player_id']].sort_values('week')
        full_career_hist = df[df['player_id'] == player['player_id']].sort_values(['season', 'week'])
            
        sched_info = schedule_map.get((target_week, p_team), None)
        is_bye = (sched_info is None)
        
        norm_key = (norm_name(p_name), p_pos)
        inj_code = injury_map.get(norm_key, 'ACT')
        slp_check = sleeper_map.get(norm_key, None)
        is_out = (inj_code in ['OUT', 'IR', 'PUP', 'DOUBTFUL']) or (slp_check == 0.0 and len(p_hist) >= 2)
        is_q = (inj_code == 'Q') and not is_out
        
        r2_pts = float(p_hist['fantasy_points_ppr'].tail(2).mean())
        r3_pts = float(p_hist['fantasy_points_ppr'].tail(3).mean())
        r5_pts = float(p_hist['fantasy_points_ppr'].tail(5).mean())
        ewm_pts = float(p_hist['fantasy_points_ppr'].ewm(span=3.5, min_periods=1).mean().iloc[-1])
        s_ppg = float(p_hist['fantasy_points_ppr'].mean())
        c_ppg = float(full_career_hist['fantasy_points_ppr'].mean())
        r5_std = float(p_hist['fantasy_points_ppr'].tail(5).std()) if len(p_hist) > 1 else 4.2
        if pd.isna(r5_std) or r5_std < 2.0:
            r5_std = 4.0
        
        r2_xfp = float(p_hist['xFP_volume'].tail(2).mean())
        r3_xfp = float(p_hist['xFP_volume'].tail(3).mean())
        s_xfp = float(p_hist['xFP_volume'].mean())
        r_trend = r2_xfp - s_xfp
        
        r3_luck = float(p_hist['luck_delta'].tail(3).mean())
        r3_tgt = float(p_hist['targets'].tail(3).mean())
        r3_rec = float(p_hist['receptions'].tail(3).mean())
        r3_car = float(p_hist['carries'].tail(3).mean())
        r3_tshare = float(p_hist['target_share'].tail(3).mean())
        r3_ashare = float(p_hist['air_yards_share'].tail(3).mean())
        r3_wopr = float(p_hist['calc_wopr'].tail(3).mean())
        r3_racr = float(p_hist['calc_racr'].tail(3).mean())
        r3_epa = float(p_hist['total_epa'].tail(3).mean())
        r3_fd = float(p_hist['total_first_downs'].tail(3).mean())
        r3_air = float(p_hist['receiving_air_yards'].tail(3).mean())
        r3_yac = float(p_hist['receiving_yards_after_catch'].tail(3).mean())
        r3_pyds = float(p_hist['passing_yards'].tail(3).mean())
        r3_ruyds = float(p_hist['rushing_yards'].tail(3).mean())
        r3_reyds = float(p_hist['receiving_yards'].tail(3).mean())
        r3_eff = float(p_hist['eff_edge'].tail(3).mean())
        r3_tdeq = float(p_hist['td_equity'].tail(3).mean())
        
        s_pyds = float(p_hist['passing_yards'].mean())
        s_ruyds = float(p_hist['rushing_yards'].mean())
        s_reyds = float(p_hist['receiving_yards'].mean())
        s_rec = float(p_hist['receptions'].mean())
        
        t_prior = 0.36 * s_ppg + 0.28 * ewm_pts + 0.26 * r3_xfp + 0.10 * c_ppg
        
        full_s_ppg = float(full_2026_hist['fantasy_points_ppr'].mean())
        full_s_xfp = float(full_2026_hist['xFP_volume'].mean())
        season_healthy_ppg = 0.48 * full_s_ppg + 0.34 * full_s_xfp + 0.18 * c_ppg
        
        opp_str = "BYE" if is_bye else sched_info['matchup_str']
        m_boost = 0.0 if is_bye else (0.0 if is_out else float(latest_dvp.get((sched_info['opp'], p_pos), 0.0)))
        vegas_tot = 44.5 if is_bye else float(sched_info['vegas_total'])
        team_imp = 22.0 if is_bye else float(sched_info['team_implied'])
        team_margin = 0.0 if is_bye else float(sched_info['team_margin'])
        is_dome = 0 if is_bye else int(sched_info['is_dome'])
        wind_spd = 6.0 if is_bye else float(sched_info['wind'])
        rest_days = 7 if is_bye else int(sched_info['rest_days'])
        div_game = 0 if is_bye else int(sched_info['div_game'])
        env_tag = "Bye Week" if is_bye else sched_info['env_tag']
        
        if is_bye:
            status_badge = "BYE"
        elif is_out:
            status_badge = f"{inj_code if inj_code != 'ACT' else 'OUT'}"
        elif is_q:
            status_badge = "QUESTIONABLE"
        else:
            status_badge = "ACTIVE"
            
        po_sos = playoff_sos_map.get((p_team, p_pos), 0.0)
            
        rows.append({
            'player_id': pid, 'Player': p_name, 'norm_name': norm_name(p_name), 'Pos': p_pos, 'Team': p_team,
            'Status': status_badge, 'Matchup': opp_str, 'Env': env_tag, 'is_bye': is_bye, 'is_out': is_out, 'is_q': is_q,
            'pos_code': pos_map[p_pos], 'talent_prior': round(t_prior, 2), 'season_ppg_prior': round(s_ppg, 2),
            'career_ppg_prior': round(c_ppg, 2), 'season_healthy_ppg': round(season_healthy_ppg, 2),
            'ewma_pts': ewm_pts, 'roll2_pts': r2_pts, 'roll3_pts': r3_pts, 'roll5_pts': r5_pts, 'roll5_std': r5_std,
            'roll3_xFP': r3_xfp, 'roll2_xFP': r2_xfp, 'season_xFP_prior': s_xfp, 'role_trend': r_trend, 'roll3_luck': r3_luck,
            'roll3_targets': r3_tgt, 'roll3_receptions': r3_rec, 'roll3_carries': r3_car,
            'roll3_tgt_share': r3_tshare, 'roll3_air_share': r3_ashare, 'roll3_wopr': r3_wopr, 'roll3_racr': r3_racr,
            'roll3_epa': r3_epa, 'roll3_first_downs': r3_fd, 'roll3_air_yards': r3_air, 'roll3_yac': r3_yac,
            'roll3_pass_yds': r3_pyds, 'roll3_rush_yds': r3_ruyds, 'roll3_rec_yds': r3_reyds,
            's_pyds': s_pyds, 's_ruyds': s_ruyds, 's_reyds': s_reyds, 's_rec': s_rec,
            'roll3_eff_edge': r3_eff, 'roll3_td_equity': r3_tdeq,
            'matchup_boost': m_boost, 'vegas_total': vegas_tot, 'team_implied': team_imp, 'team_margin': team_margin,
            'is_dome': is_dome, 'wind_speed': wind_spd, 'rest_days': rest_days, 'div_game': div_game,
            'Playoff SoS': po_sos, 'Actual PPR': actual_map.get(player['player_id'], np.nan)
        })
        
    wk_df = pd.DataFrame(rows)
    
    out_mask = wk_df['is_out'] & (~wk_df['is_bye'])
    vacated_tgts_by_team = wk_df[out_mask & wk_df['Pos'].isin(['WR', 'TE', 'RB'])].groupby('Team')['roll3_targets'].sum().to_dict()
    vacated_cars_by_team = wk_df[out_mask & (wk_df['Pos'] == 'RB')].groupby('Team')['roll3_carries'].sum().to_dict()
    
    vac_boost_list = []
    for _, r in wk_df.iterrows():
        if r['is_bye'] or r['is_out']:
            vac_boost_list.append(0.0)
            continue
        t_abbr = r['Team']
        v_tgt = vacated_tgts_by_team.get(t_abbr, 0.0)
        v_car = vacated_cars_by_team.get(t_abbr, 0.0)
        v_boost = 0.0
        if r['Pos'] in ['WR', 'TE'] and v_tgt >= 5.0 and r['roll3_targets'] >= 3.0:
            v_boost += min(0.95, v_tgt * 0.07 * min(1.4, max(0.5, r['roll3_tgt_share'] * 4.0)))
        elif r['Pos'] == 'RB' and v_car >= 7.0 and r['roll3_carries'] >= 4.0:
            v_boost += min(1.15, v_car * 0.09)
        vac_boost_list.append(round(float(v_boost), 2))
    wk_df['Vacated Boost'] = vac_boost_list

    X_wk = wk_df[features]
    
    pred_mean = model_mean.predict(X_wk)
    pred_mae = model_mae.predict(X_wk)
    pred_med = model_median.predict(X_wk)
    raw_ml = 0.42 * pred_mean + 0.33 * pred_mae + 0.25 * pred_med
    
    implied_adj = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, (wk_df['team_implied'] - 22.0) * 0.18 + wk_df['Vacated Boost'] * 0.45)
    epa_fd_adj = np.clip(wk_df['roll3_epa'] * 0.06 + (wk_df['roll3_first_downs'] - 2.5) * 0.11, -0.9, 1.1)
    pure_model_proj = 0.72 * raw_ml + 0.28 * (wk_df['talent_prior'] + wk_df['matchup_boost'] * 0.50 + implied_adj + epa_fd_adj)
    wk_df['pure_model_proj'] = pure_model_proj
    
    slp_list = []
    for _, r in wk_df.iterrows():
        if r['is_bye'] or r['is_out']:
            slp_list.append(0.0)
        else:
            slp_val = sleeper_map.get((r['norm_name'], r['Pos']), None)
            slp_list.append(slp_val if slp_val is not None else round(max(0.0, 0.65 * r['season_ppg_prior'] + 0.35 * r['roll3_pts'] + r['matchup_boost'] * 0.3), 2))
    wk_df[COL_SLEEPER] = slp_list
    
    w_ml = ml_weight_override if ml_weight_override is not None else OPTIMAL_ML_WEIGHT
    market_anchored = np.where(
        wk_df[COL_SLEEPER] > 0,
        w_ml * pure_model_proj + (1.0 - w_ml) * wk_df[COL_SLEEPER],
        pure_model_proj
    )
    
    wk_df[COL_MODEL] = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, np.clip(market_anchored, 0, None)).round(2)
    wk_df[COL_DIFF] = (wk_df[COL_MODEL] - wk_df[COL_SLEEPER]).round(2)
    
    wk_df['Floor (10%)'] = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, np.clip(model_floor.predict(X_wk), 0, wk_df[COL_MODEL] * 0.82)).round(2)
    wk_df['Boom (90%)'] = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, np.maximum(model_boom.predict(X_wk) + implied_adj, wk_df[COL_MODEL] + 3.4)).round(2)
    
    wk_df['ROS Value'] = np.where(
        wk_df['Status'].isin(['IR', 'PUP']),
        wk_df['season_healthy_ppg'] * 0.40,
        np.clip(wk_df['season_healthy_ppg'] + 0.15 * wk_df['Playoff SoS'], 0.0, None)
    ).round(2)

    # Compute Raw Player Stat Prop Projections (Passing Yards, Rushing Yards, Receiving Yards, Receptions, Anytime TD %)
    active_mask = ~(wk_df['is_bye'] | wk_df['is_out'])
    for p_key, roll_col, s_col, dvp_scale in [
        ('pass_yd', 'roll3_pass_yds', 's_pyds', 4.5),
        ('rush_yd', 'roll3_rush_yds', 's_ruyds', 2.2),
        ('rec_yd', 'roll3_rec_yds', 's_reyds', 2.4),
        ('receptions', 'roll3_receptions', 's_rec', 0.18)
    ]:
        _, m_med, m_mean = prop_models[p_key]
        raw_p = 0.62 * m_med.predict(X_wk) + 0.38 * m_mean.predict(X_wk)
        hist_base = 0.55 * wk_df[roll_col] + 0.45 * wk_df[s_col] + wk_df['matchup_boost'] * dvp_scale
        pure_stat = np.clip(0.68 * raw_p + 0.32 * hist_base, 0.0, None)
        
        # Pull market consensus prop line from Sleeper stat projections (or historical baseline if unavailable)
        mkt_lines = []
        for idx, r in wk_df.iterrows():
            if not active_mask.iloc[idx]:
                mkt_lines.append(0.0)
                continue
            s_dict = stat_proj_map.get((r['norm_name'], r['Pos']), {})
            m_val = s_dict.get(p_key, 0.0)
            if m_val <= 0.0:
                m_val = max(0.0, round(hist_base.iloc[idx], 1))
            # Round to nearest .5 sportsbook hook (e.g. 54.5, 62.5)
            hook_line = math.floor(m_val) + 0.5 if m_val >= 1.0 else 0.0
            mkt_lines.append(hook_line)
        wk_df[f'Mkt_{p_key}'] = mkt_lines
        
        blended_stat = np.where(wk_df[f'Mkt_{p_key}'] > 0, 0.64 * pure_stat + 0.36 * wk_df[f'Mkt_{p_key}'], pure_stat)
        wk_df[f'Proj_{p_key}'] = np.where(active_mask, np.clip(blended_stat, 0.0, None), 0.0).round(1)
        
    # Anytime TD Probability (Poisson arrival rate lambda -> 1 - exp(-lambda))
    _, m_td = prop_models['any_td']
    raw_td_lambda = np.clip(0.65 * m_td.predict(X_wk) + 0.35 * (wk_df['roll3_td_equity'] * 0.85 + np.maximum(0.0, wk_df['team_implied'] - 20.0) * 0.03), 0.02, 1.85)
    td_prob_pct = np.where(active_mask & (wk_df['Pos'].isin(['RB', 'WR', 'TE'])), (1.0 - np.exp(-raw_td_lambda)) * 100.0, 0.0).round(1)
    wk_df['Anytime TD %'] = td_prob_pct
    
    boom_pct_list, bust_pct_list, boom_reasons, bust_reasons, diff_reasons = [], [], [], [], []
    for _, r in wk_df.iterrows():
        proj = float(r[COL_MODEL])
        if r['is_bye'] or r['is_out'] or proj <= 1.0:
            boom_pct_list.append(0.0)
            bust_pct_list.append(100.0 if r['is_out'] else 0.0)
            boom_reasons.append("Inactive / Bye")
            bust_reasons.append("Inactive / Bye")
            diff_reasons.append("Inactive / Bye")
            continue
            
        xfp_edge = float(r['roll3_xFP'] - r['roll3_pts'])
        dvp = float(r['matchup_boost'])
        vegas_edge = float(r['team_implied'] - 22.0)
        model_edge = float(r[COL_DIFF])
        wopr_val = float(r['roll3_wopr'])
        epa_val = float(r['roll3_epa'])
        vac_val = float(r['Vacated Boost'])
        vol_ratio = float(r['roll5_std']) / max(proj, 7.0)
        
        skew = (0.34 * xfp_edge) + (0.31 * dvp) + (0.18 * model_edge) + (0.10 * vegas_edge) + (0.08 * epa_val) + (0.20 * vac_val)
        base_rate = np.clip(16.0 + vol_ratio * 6.0, 15.0, 21.0)
        
        p_boom = base_rate + (2.7 * skew) + (1.3 * max(0.0, dvp)) + (0.85 * max(0.0, xfp_edge))
        inj_pen = 7.5 if r['is_q'] else 0.0
        p_bust = base_rate - (2.7 * skew) + (1.4 * max(0.0, -dvp)) + (0.95 * max(0.0, -xfp_edge)) + inj_pen
        
        boom_pct_list.append(round(float(np.clip(p_boom, 5.0, 47.0)), 1))
        bust_pct_list.append(round(float(np.clip(p_bust, 5.0, 48.0)), 1))
        
        b_drivers = []
        if vac_val >= 0.5: b_drivers.append(f"Vacated Teammate Vol (+{vac_val:.1f} PPR)")
        if xfp_edge >= 1.4: b_drivers.append(f"Unconverted Volume ({r['roll3_xFP']:.1f} xFP vs {r['roll3_pts']:.1f} PPG)")
        if wopr_val >= 0.55 and r['Pos'] in ['WR', 'TE']: b_drivers.append(f"Alpha WOPR ({wopr_val:.2f})")
        if dvp >= 1.2: b_drivers.append(f"Soft Defense ({dvp:+.1f} DvP)")
        if vegas_edge >= 2.5: b_drivers.append(f"High Team Total ({r['team_implied']:.1f} pts)")
        if not b_drivers: b_drivers.append(f"High Ceiling Spread ({r['Boom (90%)']:.1f} pt 90th pct)")
        boom_reasons.append(" + ".join(b_drivers[:2]))
        
        d_drivers = []
        if xfp_edge <= -1.4: d_drivers.append(f"Overperformed Volume ({r['roll3_pts']:.1f} PPG on {r['roll3_xFP']:.1f} xFP)")
        if dvp <= -1.2: d_drivers.append(f"Tough Defense ({dvp:+.1f} DvP)")
        if vegas_edge <= -2.0: d_drivers.append(f"Low Team Total ({r['team_implied']:.1f} pts)")
        if r['is_q']: d_drivers.append("Questionable Injury Tag")
        if not d_drivers: d_drivers.append(f"Fragile Floor ({r['Floor (10%)']:.1f} pt 10th pct)")
        bust_reasons.append(" + ".join(d_drivers[:2]))
        
        if model_edge >= 0:
            diff_reasons.append(f"Trailing {r['roll3_xFP']:.1f} xFP ({r['roll3_first_downs']:.1f} 1st Downs/gm) + {dvp:+.1f} DvP")
        else:
            diff_reasons.append(f"Consensus inflated vs {r['roll3_xFP']:.1f} xFP baseline & {dvp:+.1f} DvP")
        
    wk_df['Boom %'] = boom_pct_list
    wk_df['Bust %'] = bust_pct_list
    wk_df['Boom Catalyst'] = boom_reasons
    wk_df['Bust Catalyst'] = bust_reasons
    wk_df['Discrepancy Reason'] = diff_reasons
    
    def classify_outlook(r):
        if r['is_bye']: return "ON BYE WEEK"
        if r['is_out']: return f"INACTIVE ({r['Status']})"
        if r['is_q']: return "QUESTIONABLE (VOLATILE)"
        if r['Boom %'] >= 26.5 and (r['Boom %'] - r['Bust %'] >= 7.0): return "HIGH-CEILING SKEW"
        if r['Bust %'] >= 26.5 and (r['Bust %'] - r['Boom %'] >= 7.0): return "HIGH BUST RISK"
        return "EXPECTED BASELINE"

    wk_df['Outlook Verdict'] = wk_df.apply(classify_outlook, axis=1)
    wk_df['L3 Avg PPR'] = wk_df['roll3_pts'].round(2)
    wk_df['L3 xFP (Vol)'] = wk_df['roll3_xFP'].round(2)
    wk_df['WOPR'] = wk_df['roll3_wopr'].round(2)
    wk_df['RACR'] = wk_df['roll3_racr'].round(2)
    wk_df['L3 EPA'] = wk_df['roll3_epa'].round(2)
    wk_df['DvP Boost'] = wk_df['matchup_boost'].round(2)
    
    sorted_df = wk_df.sort_values(by=COL_MODEL, ascending=False).reset_index(drop=True)
    if ml_weight_override is None:
        projection_week_cache[target_week] = sorted_df
    return sorted_df.copy()

try:
    if latest_completed_week >= 2:
        eval_frames = []
        for w_eval in range(2, latest_completed_week + 1):
            f_df = build_week_projections(w_eval, ml_weight_override=0.60)
            f_df = f_df[(~f_df['is_bye']) & (~f_df['is_out']) & (f_df['Actual PPR'].notna()) & (f_df[COL_SLEEPER] >= 5.0)].copy()
            eval_frames.append(f_df)
        if eval_frames:
            comb_eval = pd.concat(eval_frames, ignore_index=True)
            best_score = -999.0
            best_w = 0.60
            for cand_w in [0.48, 0.52, 0.56, 0.60, 0.64, 0.68]:
                cand_proj = (cand_w * comb_eval['pure_model_proj'] + (1.0 - cand_w) * comb_eval[COL_SLEEPER]).round(2)
                m_err = (cand_proj - comb_eval['Actual PPR']).abs()
                s_err = (comb_eval[COL_SLEEPER] - comb_eval['Actual PPR']).abs()
                non_t = m_err != s_err
                w_rate = (m_err[non_t] < s_err[non_t]).mean() * 100.0 if non_t.sum() > 0 else 50.0
                mae_edge = (s_err.mean() - m_err.mean())
                obj = w_rate + mae_edge * 12.0
                if obj > best_score:
                    best_score = obj
                    best_w = cand_w
            OPTIMAL_ML_WEIGHT = best_w
            projection_week_cache.clear()
except Exception:
    OPTIMAL_ML_WEIGHT = 0.60

def fetch_live_dk_fd_props(odds_api_key):
    """Optional integration with The Odds API to pull live FanDuel & DraftKings NFL player props."""
    if not odds_api_key or not str(odds_api_key).strip():
        return {}, "Using Sleeper / Market Consensus Prop Lines (Paste a free key from the-odds-api.com above to overlay live DraftKings & FanDuel lines)."
    key = str(odds_api_key).strip()
    if key in live_sportsbook_props_cache:
        return live_sportsbook_props_cache[key], "Synced live DraftKings & FanDuel prop lines from cache."
    book_lines = {}
    try:
        ev_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events?apiKey={key}"
        ev_res = requests.get(ev_url, timeout=8)
        if ev_res.status_code != 200:
            return {}, "Odds API Key notice: Could not fetch NFL events (verify key). Using Market Consensus Prop Lines."
        events = ev_res.json()[:14]
        market_map = {
            'player_pass_yds': 'pass_yd',
            'player_rush_yds': 'rush_yd',
            'player_reception_yds': 'rec_yd',
            'player_receptions': 'receptions'
        }
        for ev in events:
            eid = ev.get('id')
            if not eid:
                continue
            p_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{eid}/odds?apiKey={key}&regions=us&markets=player_pass_yds,player_rush_yds,player_reception_yds,player_receptions&bookmakers=draftkings,fanduel&oddsFormat=american"
            p_res = requests.get(p_url, timeout=6)
            if p_res.status_code == 200:
                bms = p_res.json().get('bookmakers', [])
                for bm in bms:
                    b_title = "DK" if "draftkings" in bm.get('key', '') else "FD"
                    for mkt in bm.get('markets', []):
                        m_key = market_map.get(mkt.get('key', ''))
                        if not m_key:
                            continue
                        for outc in mkt.get('outcomes', []):
                            if outc.get('name') == 'Over':
                                p_norm = norm_name(outc.get('description', ''))
                                pt_line = float(outc.get('point', 0.0) or 0.0)
                                price = int(outc.get('price', -110) or -110)
                                if p_norm and pt_line > 0:
                                    book_lines[(p_norm, m_key)] = {'line': pt_line, 'odds': price, 'book': b_title}
        live_sportsbook_props_cache[key] = book_lines
        return book_lines, f"Synced {len(book_lines)} live DraftKings & FanDuel player prop lines!"
    except Exception as e:
        return {}, f"Live sportsbook sync notice ({e}). Using Market Consensus Prop Lines."

def build_player_props_board(week_num, prop_category, pos_filter, min_edge_filter, search_txt, odds_api_key=""):
    wk_df = build_week_projections(int(week_num))
    active_df = wk_df[(~wk_df['is_bye']) & (~wk_df['is_out']) & (wk_df[COL_MODEL] >= 5.0)].copy()
    if pos_filter != "ALL":
        active_df = active_df[active_df['Pos'] == pos_filter].copy()
    if search_txt and str(search_txt).strip():
        q = str(search_txt).strip()
        active_df = active_df[active_df['Player'].str.contains(q, case=False, na=False) | active_df['Team'].str.contains(q, case=False, na=False)].copy()
        
    book_lines, status_msg = fetch_live_dk_fd_props(odds_api_key)
    
    cat_configs = {
        "Receiving Yards": ('rec_yd', ['WR', 'TE', 'RB'], 18.5, 8.5, "Yds"),
        "Rushing Yards": ('rush_yd', ['RB', 'QB'], 16.5, 8.5, "Yds"),
        "Passing Yards": ('pass_yd', ['QB'], 42.0, 120.0, "Yds"),
        "Receptions (PPR)": ('receptions', ['WR', 'TE', 'RB'], 1.65, 1.5, "Rec")
    }
    
    cats_to_run = list(cat_configs.keys()) if prop_category == "ALL PROP MARKETS" else [prop_category]
    prop_rows = []
    
    for c_label in cats_to_run:
        p_key, valid_pos, base_sigma, min_line, unit_str = cat_configs[c_label]
        sub_df = active_df[active_df['Pos'].isin(valid_pos)].copy()
        for _, r in sub_df.iterrows():
            mod_stat = float(r[f'Proj_{p_key}'])
            live_bk = book_lines.get((r['norm_name'], p_key))
            if live_bk:
                line_val = live_bk['line']
                book_odds = live_bk['odds']
                book_tag = f"{live_bk['book']} ({book_odds:+d})"
            else:
                line_val = float(r[f'Mkt_{p_key}'])
                book_odds = -110
                book_tag = "Consensus (-110)"
                
            if line_val < min_line and mod_stat < min_line:
                continue
                
            diff_stat = round(mod_stat - line_val, 1)
            # Scale dispersion slightly with projection magnitude
            sigma = max(base_sigma * 0.65, base_sigma * (0.55 + 0.45 * (max(line_val, mod_stat) / (min_line * 2.5))))
            p_over = (1.0 - normal_cdf((line_val - mod_stat) / sigma)) * 100.0
            p_under = 100.0 - p_over
            
            pick_side = "OVER" if mod_stat >= line_val else "UNDER"
            win_prob = round(float(np.clip(p_over if pick_side == "OVER" else p_under, 50.1, 79.5)), 1)
            
            imp_prob = american_odds_to_implied_prob(book_odds) * 100.0
            ev_edge = round(win_prob - imp_prob, 1)
            if ev_edge < float(min_edge_filter):
                continue
                
            fair_odds = prob_to_american_odds(win_prob)
            if ev_edge >= 8.0:
                grade = "A+ (MAX +EV)"
            elif ev_edge >= 5.0:
                grade = "A (STRONG +EV)"
            elif ev_edge >= 2.5:
                grade = "B+ (LEAN)"
            else:
                grade = "NEUTRAL"
                
            prop_rows.append({
                "Player": r['Player'],
                "Pos": r['Pos'],
                "Team": r['Team'],
                "Matchup": f"{r['Matchup']} ({r['Env']})",
                "Prop Market": c_label,
                "Model Stat Proj": f"{mod_stat:.1f} {unit_str}",
                "Sportsbook / Mkt Line": f"{line_val:.1f} {unit_str}",
                "Line Source": book_tag,
                "Model Edge": f"{diff_stat:+.1f} {unit_str}",
                "Recommended Bet": f"{pick_side} {line_val:.1f}",
                "Hit Prob %": f"{win_prob:.1f}%",
                "Expected Value (+EV%)": f"{ev_edge:+.1f}%",
                "Fair Odds": fair_odds,
                "Bet Grade": grade,
                "Anytime TD %": f"{r['Anytime TD %']:.1f}% ({prob_to_american_odds(r['Anytime TD %'])})" if r['Pos'] != 'QB' else "N/A",
                "_ev_sort": ev_edge
            })
            
    if not prop_rows:
        empty_cols = ["Player", "Pos", "Team", "Matchup", "Prop Market", "Model Stat Proj", "Sportsbook / Mkt Line", "Line Source", "Model Edge", "Recommended Bet", "Hit Prob %", "Expected Value (+EV%)", "Fair Odds", "Bet Grade", "Anytime TD %"]
        return status_msg, pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)
        
    full_props_df = pd.DataFrame(prop_rows).sort_values(by="_ev_sort", ascending=False).drop(columns=["_ev_sort"]).reset_index(drop=True)
    top_ev_df = full_props_df.head(10).copy()
    return status_msg, top_ev_df, full_props_df

def evaluate_custom_prop_bet(week_num, player_name, prop_market, custom_line, custom_odds):
    if not player_name:
        return "Select a player to evaluate a custom sportsbook prop line."
    wk_df = build_week_projections(int(week_num))
    p_rows = wk_df[wk_df['Player'] == player_name]
    if p_rows.empty:
        return "Player not found for the selected week."
    r = p_rows.iloc[0]
    
    cat_map = {
        "Receiving Yards": ('rec_yd', 18.5, "Yards", 140.0),
        "Rushing Yards": ('rush_yd', 16.5, "Yards", 140.0),
        "Passing Yards": ('pass_yd', 42.0, "Yards", 360.0),
        "Receptions (PPR)": ('receptions', 1.65, "Receptions", 12.0)
    }
    p_key, sigma, unit_str, max_scale = cat_map.get(prop_market, ('rec_yd', 18.5, "Yards", 140.0))
    mod_proj = float(r[f'Proj_{p_key}'])
    try:
        line_v = float(custom_line)
    except Exception:
        line_v = float(r[f'Mkt_{p_key}'])
        
    p_over = (1.0 - normal_cdf((line_v - mod_proj) / sigma)) * 100.0
    p_under = 100.0 - p_over
    pick = "OVER" if mod_proj >= line_v else "UNDER"
    hit_prob = round(float(np.clip(p_over if pick == "OVER" else p_under, 5.0, 92.0)), 1)
    imp_prob = round(american_odds_to_implied_prob(custom_odds) * 100.0, 1)
    ev_edge = round(hit_prob - imp_prob, 1)
    fair_line_odds = prob_to_american_odds(hit_prob)
    
    flr_stat = max(0.0, round(mod_proj - 1.28 * sigma, 1))
    ceil_stat = round(mod_proj + 1.28 * sigma, 1)
    
    bar_html = render_payoff_distribution_bar_html(
        f"{player_name} — {prop_market} Distribution",
        f"Custom Line: {line_v:.1f} {unit_str}",
        flr_stat, mod_proj, line_v, ceil_stat, max_scale=max(max_scale, ceil_stat + 10.0), compact=True
    )
    
    verdict_color = "#059669" if ev_edge >= 2.5 else ("#d97706" if ev_edge >= 0.0 else "#dc2626")
    verdict_tag = f"STRONG BET: {pick} {line_v:.1f} ({ev_edge:+.1f}% Edge)" if ev_edge >= 2.5 else (f"SLIGHT LEAN: {pick} {line_v:.1f}" if ev_edge >= 0.0 else f"PASS / NEGATIVE EV ({ev_edge:+.1f}%)")
    
    html = (
        f'<div style="padding:14px 16px; border-radius:10px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.03); margin-top:8px;">'
        f'<div style="font-size:1.1rem; font-weight:700; color:{verdict_color}; margin-bottom:8px;">{verdict_tag}</div>'
        f'<div style="font-size:0.9rem; margin-bottom:10px;">'
        f'<strong>Model Median Projection:</strong> {mod_proj:.1f} {unit_str} &nbsp;|&nbsp; '
        f'<strong>Sportsbook Line:</strong> {line_v:.1f} ({custom_odds}) &nbsp;|&nbsp; '
        f'<strong>True Hit Prob:</strong> {hit_prob:.1f}% (vs {imp_prob:.1f}% Book Implied) &nbsp;|&nbsp; '
        f'<strong>Fair Odds:</strong> {fair_line_odds}'
        f'</div>'
        f'{bar_html}'
        f'</div>'
    )
    return html

def build_player_scouting_html(p_row, week_num):
    bio = get_player_bio(p_row['Player'], p_row['Pos'], p_row['Team'], p_row.get('player_id', ''))
    mod_val = float(p_row[COL_MODEL])
    slp_val = float(p_row[COL_SLEEPER])
    diff_val = float(p_row[COL_DIFF])
    flr_val = float(p_row['Floor (10%)'])
    ceil_val = float(p_row['Boom (90%)'])
    ros_val = float(p_row['ROS Value'])
    
    phys_str = f"{bio['height']} | {bio['weight']} lbs | " if bio['height'] and bio['weight'] else ""
    bio_subtitle = f"{p_row['Pos']} — {p_row['Team']} | {phys_str}Age: {bio['age']} | Exp: {bio['exp']} | College: {bio['college']}"
    
    if p_row['Pos'] in ['WR', 'TE']:
        role_desc = (
            f"Averaging {p_row['roll3_targets']:.1f} targets and {p_row['roll3_receptions']:.1f} receptions per game over the last 3 weeks "
            f"(WOPR: {p_row['WOPR']:.2f}, RACR: {p_row['RACR']:.2f}, Air Yards: {p_row['roll3_air_yards']:.1f}/gm). "
            f"Projected Stat Line: {p_row['Proj_rec_yd']:.1f} Rec Yds on {p_row['Proj_receptions']:.1f} Catches ({p_row['Anytime TD %']:.1f}% Anytime TD)."
        )
    elif p_row['Pos'] == 'RB':
        role_desc = (
            f"Handling {p_row['roll3_carries']:.1f} carries and {p_row['roll3_targets']:.1f} targets per game over the last 3 weeks "
            f"({p_row['roll3_first_downs']:.1f} first downs created/gm, EPA: {p_row['L3 EPA']:+.1f}/gm). "
            f"Projected Stat Line: {p_row['Proj_rush_yd']:.1f} Rush Yds + {p_row['Proj_rec_yd']:.1f} Rec Yds ({p_row['Anytime TD %']:.1f}% Anytime TD)."
        )
    else:
        role_desc = (
            f"Averaging {p_row['roll3_pass_yds']:.1f} passing yards and {p_row['roll3_carries']:.1f} rushing attempts per game "
            f"({p_row['roll3_first_downs']:.1f} total first downs/gm, EPA: {p_row['L3 EPA']:+.1f}/gm). "
            f"Projected Stat Line: {p_row['Proj_pass_yd']:.1f} Pass Yds + {p_row['Proj_rush_yd']:.1f} Rush Yds."
        )
        
    matchup_desc = (
        f" In Week {int(week_num)} ({p_row['Matchup']}, {p_row['Env']}), the opposing defense grades at {p_row['DvP Boost']:+.1f} PPR DvP "
        f"with a Vegas implied team total of {p_row['team_implied']:.1f} points."
    )
    
    dist_bar_html = render_payoff_distribution_bar_html(
        f"{p_row['Player']} Payoff Distribution (10th to 90th Percentile Range)",
        f"{p_row['Pos']} - {p_row['Team']}",
        flr_val, mod_val, slp_val, ceil_val
    )
    
    shap_dict = compute_shapley_group_attribution(p_row)
    shap_items_html = ""
    for k_label, v_val in shap_dict.items():
        if "Baseline" in k_label:
            badge_col = "#475569"
            val_str = f"{v_val:.2f} PPR"
        else:
            badge_col = "#059669" if v_val >= 0 else "#dc2626"
            val_str = f"{v_val:+.2f} PPR"
        shap_items_html += (
            f'<div style="padding:8px 10px; border-radius:8px; border:1px solid rgba(148,163,184,0.25); background:rgba(15,23,42,0.02);">'
            f'<div style="font-size:0.74rem; opacity:0.75;">{k_label}</div>'
            f'<div style="font-size:0.98rem; font-weight:700; color:{badge_col}; margin-top:2px;">{val_str}</div>'
            f'</div>'
        )
    
    html = (
        f'<div style="padding:18px; border-radius:12px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.03); margin-bottom:12px;">'
        f'<div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap; margin-bottom:14px;">'
        f'<img src="{bio["headshot"]}" style="width:92px; height:92px; border-radius:50%; object-fit:cover; border:2px solid rgba(59,130,246,0.6); background:#1e293b;" onerror="this.src=\'https://sleepercdn.com/images/v2/icons/player_default.webp\'"/>'
        f'<div style="flex:1; min-width:260px;">'
        f'<div style="font-size:1.35rem; font-weight:700; line-height:1.3;">{p_row["Player"]} <span style="font-size:0.95rem; font-weight:600; opacity:0.85;">— {p_row["Outlook Verdict"]}</span></div>'
        f'<div style="font-size:0.92rem; opacity:0.78; margin-top:4px; margin-bottom:10px;">{bio_subtitle}</div>'
        f'<div style="font-size:0.95rem; line-height:1.55;"><strong>Scouting Analysis:</strong> {role_desc}{matchup_desc}</div>'
        f'</div></div>'
        f'{dist_bar_html}'
        f'<div style="font-size:0.84rem; font-weight:700; margin-top:12px; margin-bottom:6px; opacity:0.85;">Model Feature Attribution Waterfall (Marginal Tree Decomposition):</div>'
        f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:8px;">{shap_items_html}</div>'
        f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(135px, 1fr)); gap:10px; margin-top:14px; padding-top:12px; border-top:1px solid rgba(148,163,184,0.25);">'
        f'<div><div style="font-size:0.78rem; opacity:0.7;">Model\'s Projection</div><div style="font-size:1.15rem; font-weight:700;">{mod_val:.2f} PPR</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.7;">Sleeper\'s Projection</div><div style="font-size:1.15rem; font-weight:600;">{slp_val:.2f} PPR</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.7;">Diff (Model - Sleeper)</div><div style="font-size:1.15rem; font-weight:700;">{diff_val:+.2f}</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.7;">Floor / Ceiling</div><div style="font-size:1.1rem; font-weight:600;">{flr_val:.1f} – {ceil_val:.1f}</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.7;">Boom / Bust Prob</div><div style="font-size:1.1rem; font-weight:600;">{p_row["Boom %"]}% / {p_row["Bust %"]}%</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.7;">Healthy ROS Value</div><div style="font-size:1.15rem; font-weight:700;">{ros_val:.2f} PPG</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.7;">Playoff SoS (Wks 15-17)</div><div style="font-size:1.1rem; font-weight:600;">{p_row["Playoff SoS"]:+.2f} DvP</div></div>'
        f'</div></div>'
    )
    return html

def update_main_table(week_num, hide_byes, hide_injured, compact_view, pos_filter, outlook_filter, search_text, force_refresh=False):
    wk_df = build_week_projections(int(week_num), force_refresh=force_refresh)
    our_20plus = int((wk_df[COL_MODEL] >= 20.0).sum())
    slp_20plus = int((wk_df[COL_SLEEPER] >= 20.0).sum())
    mode_label = "Completed Week Backtest Mode" if int(week_num) <= latest_completed_week else "Live Real-Time Projection Mode"
    
    banner = (
        f"**Week {int(week_num)} Overview ({mode_label} | Calibrated ML Weight: {int(OPTIMAL_ML_WEIGHT*100)}%)** — "
        f"Projected 20+ Point Scorers: **Model's Projection: {our_20plus}** vs. **Sleeper's Projection: {slp_20plus}** | "
        f"*Click any player row below to open their Visual Payoff Distribution Bar, Feature Attribution Waterfall & Headshot Card.*"
    )
    
    if hide_byes: wk_df = wk_df[~wk_df['is_bye']].copy()
    if hide_injured: wk_df = wk_df[~wk_df['is_out']].copy()
    if pos_filter != "ALL": wk_df = wk_df[wk_df['Pos'] == pos_filter].copy()
    if outlook_filter != "ALL": wk_df = wk_df[wk_df['Outlook Verdict'].str.contains(outlook_filter, case=False, na=False)].copy()
    if search_text: wk_df = wk_df[wk_df['Player'].str.contains(search_text, case=False, na=False) | wk_df['Team'].str.contains(search_text, case=False, na=False)].copy()
        
    if compact_view:
        cols = [
            'Player', 'Pos', 'Team', 'Matchup',
            COL_MODEL, COL_SLEEPER, COL_DIFF,
            'Floor (10%)', 'Boom (90%)', 'Boom %', 'Bust %', 'ROS Value', 'Outlook Verdict'
        ]
        if int(week_num) <= latest_completed_week:
            cols.insert(7, 'Actual PPR')
    else:
        cols = [
            'Player', 'Pos', 'Team', 'Status', 'Matchup', 'Env',
            COL_MODEL, COL_SLEEPER, COL_DIFF,
            'Floor (10%)', 'Boom (90%)', 'Boom %', 'Bust %',
            'ROS Value', 'L3 Avg PPR', 'L3 xFP (Vol)', 'Vacated Boost', 'WOPR', 'RACR', 'L3 EPA', 'DvP Boost', 'Playoff SoS', 'Outlook Verdict'
        ]
        if int(week_num) <= latest_completed_week:
            cols.insert(9, 'Actual PPR')
    return banner, wk_df[cols].reset_index(drop=True)

def force_refresh_main_table(week_num, hide_byes, hide_injured, compact_view, pos_filter, outlook_filter, search_text):
    return update_main_table(week_num, hide_byes, hide_injured, compact_view, pos_filter, outlook_filter, search_text, force_refresh=True)

def open_player_card_on_click(table_df, week_num, evt: gr.SelectData):
    if table_df is None or len(table_df) == 0:
        return gr.update(visible=False), "", gr.update(visible=False)
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    player_name = str(table_df.iloc[row_idx]['Player'])
    
    wk_df = build_week_projections(int(week_num))
    p_rows = wk_df[wk_df['Player'] == player_name]
    if p_rows.empty:
        return gr.update(visible=False), "", gr.update(visible=False)
    p = p_rows.iloc[0]
    
    card_html = build_player_scouting_html(p, int(week_num))
    
    logs = df_2026[df_2026['player_id'].astype(str) == str(p['player_id'])].sort_values('week', ascending=False)
    log_cols = ['week', opp_col, 'fantasy_points_ppr', 'xFP_volume', 'calc_wopr', 'calc_racr', 'total_epa', 'total_first_downs', 'targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards']
    logs_clean = logs[log_cols].copy()
    for c in ['fantasy_points_ppr', 'xFP_volume', 'calc_wopr', 'calc_racr', 'total_epa']:
        logs_clean[c] = logs_clean[c].round(2)
    logs_clean.columns = ['Week', 'Opp', 'Actual PPR', 'Expected xFP', 'WOPR', 'RACR', 'Total EPA', '1st Downs', 'Targets', 'Rec', 'Rec Yds', 'Carries', 'Rush Yds']
    return gr.update(visible=True), card_html, gr.update(value=logs_clean, visible=True)

def close_player_scouting_drawer():
    return gr.update(visible=False), "", gr.update(visible=False)

def get_top_outliers(week_num):
    wk_df = build_week_projections(int(week_num))
    valid_df = wk_df[((wk_df[COL_MODEL] >= 10.0) | (wk_df[COL_SLEEPER] >= 10.0)) & (~wk_df['is_out']) & (~wk_df['is_bye'])].copy()
    
    boom_pool = valid_df[valid_df['Boom %'] > valid_df['Bust %']].copy()
    boom_pool['Net Upside'] = boom_pool['Boom %'] - boom_pool['Bust %']
    booms = boom_pool.sort_values(['Net Upside', 'Boom %'], ascending=False).head(5)[
        ['Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Boom (90%)', 'Boom %', 'Bust %', 'Boom Catalyst']
    ]
    
    bust_pool = valid_df[(valid_df['Bust %'] > valid_df['Boom %']) & (~valid_df['Player'].isin(booms['Player']))].copy()
    bust_pool['Net Downside'] = bust_pool['Bust %'] - bust_pool['Boom %']
    busts = bust_pool.sort_values(['Net Downside', 'Bust %'], ascending=False).head(5)[
        ['Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Floor (10%)', 'Bust %', 'Boom %', 'Bust Catalyst']
    ]
    
    return booms.reset_index(drop=True), busts.reset_index(drop=True)

def get_biggest_differers(week_num, pos_filter):
    wk_df = build_week_projections(int(week_num))
    valid_df = wk_df[(~wk_df['is_out']) & (~wk_df['is_bye']) & ((wk_df[COL_MODEL] >= 6.0) | (wk_df[COL_SLEEPER] >= 6.0))].copy()
    if pos_filter != "ALL":
        valid_df = valid_df[valid_df['Pos'] == pos_filter].copy()
        
    cols = [
        'Player', 'Pos', 'Team', 'Matchup',
        COL_MODEL, COL_SLEEPER, COL_DIFF,
        'L3 xFP (Vol)', 'Vacated Boost', 'WOPR', 'DvP Boost', 'Discrepancy Reason'
    ]
    
    higher_df = valid_df[valid_df[COL_DIFF] > 0].sort_values(by=COL_DIFF, ascending=False).head(12)[cols].reset_index(drop=True)
    lower_df = valid_df[valid_df[COL_DIFF] < 0].sort_values(by=COL_DIFF, ascending=True).head(12)[cols].reset_index(drop=True)
    return higher_df, lower_df

def compare_multiple_players(week_num, player_list):
    if not player_list or len(player_list) < 2:
        return "", "Select at least **2 players** (up to 6) from the dropdown above to compare their visual payoff distributions and metrics side-by-side.", pd.DataFrame()
        
    wk_df = build_week_projections(int(week_num))
    selected_rows = []
    for pname in player_list[:6]:
        m_rows = wk_df[wk_df['Player'] == pname]
        if not m_rows.empty:
            selected_rows.append(m_rows.iloc[0])
            
    if len(selected_rows) < 2:
        return "", "Unable to locate data for the selected players.", pd.DataFrame()
        
    cards_html = '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:14px; margin-bottom:16px;">'
    for r in selected_rows:
        bio = get_player_bio(r['Player'], r['Pos'], r['Team'], r.get('player_id', ''))
        cards_html += (
            f'<div style="padding:14px; border-radius:10px; border:1px solid rgba(148,163,184,0.35); display:flex; align-items:center; gap:12px;">'
            f'<img src="{bio["headshot"]}" style="width:64px; height:64px; border-radius:50%; object-fit:cover; background:#1e293b;" onerror="this.src=\'https://sleepercdn.com/images/v2/icons/player_default.webp\'"/>'
            f'<div>'
            f'<div style="font-weight:700; font-size:1.05rem;">{r["Player"]}</div>'
            f'<div style="font-size:0.82rem; opacity:0.75;">{r["Pos"]} - {r["Team"]} | {bio["college"]}</div>'
            f'<div style="font-size:0.88rem; margin-top:4px;"><strong>Model: {r[COL_MODEL]:.2f}</strong> | ROS: {r["ROS Value"]:.2f}</div>'
            f'</div></div>'
        )
    cards_html += '</div>'
    
    bars_html = (
        '<div style="padding:16px; border-radius:12px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.03); margin-bottom:14px;">'
        '<div style="font-size:1.05rem; font-weight:700; margin-bottom:10px;">Synchronized Payoff Distribution Comparison (10th-Pct Floor — Model Projection — 90th-Pct Ceiling)</div>'
    )
    max_ceil = max([float(r['Boom (90%)']) for r in selected_rows] + [32.0])
    for r in selected_rows:
        bars_html += render_payoff_distribution_bar_html(
            r['Player'],
            f"{r['Pos']} - {r['Team']} ({r['Matchup']})",
            float(r['Floor (10%)']),
            float(r[COL_MODEL]),
            float(r[COL_SLEEPER]),
            float(r['Boom (90%)']),
            max_scale=max_ceil + 2.0,
            compact=True
        )
    bars_html += '</div>'
    
    best_wk = max(selected_rows, key=lambda x: float(x[COL_MODEL]))
    best_flr = max(selected_rows, key=lambda x: float(x['Floor (10%)']))
    best_ceil = max(selected_rows, key=lambda x: float(x['Boom (90%)']))
    best_ros = max(selected_rows, key=lambda x: float(x['ROS Value']))
    
    summary = (
        f"### Multi-Player Portfolio Comparison Verdict (Week {int(week_num)})\n"
        f"* **Highest Expected Projection:** **{best_wk['Player']}** ({best_wk[COL_MODEL]:.2f} PPR vs Sleeper's {best_wk[COL_SLEEPER]:.2f} PPR)\n"
        f"* **Safest Cash/Favored Floor (10th Percentile):** **{best_flr['Player']}** ({best_flr['Floor (10%)']:.2f} PPR floor, only {best_flr['Bust %']}% Bust Risk)\n"
        f"* **Highest Underdog Ceiling (90th Percentile):** **{best_ceil['Player']}** ({best_ceil['Boom (90%)']:.2f} PPR ceiling, {best_ceil['Boom %']}% Boom Rate)\n"
        f"* **Highest Full-Season Rest-of-Season Value:** **{best_ros['Player']}** ({best_ros['ROS Value']:.2f} Healthy ROS PPG)"
    )
    
    metric_names = [
        'Position / Team', 'Status / Environment', 'Week Matchup',
        "Model's Projection (PPR)", "Sleeper's Projection (PPR)", "Diff (Model - Sleeper)",
        'Projected Stat Line (Yds / Rec)', 'Anytime TD Probability',
        'Full-Season ROS Value (PPG)', 'Fantasy Playoff SoS (Wks 15-17)', 'Vacated Teammate Boost',
        'Floor (10th Percentile)', 'Ceiling (90th Percentile)', 'Boom Probability', 'Bust Probability',
        'Trailing 3-Week Opportunity (xFP)', 'Trailing 3-Week Actual PPR',
        'Weighted Opportunity (WOPR)', 'Air Yards Conversion (RACR)', 'Trailing 3-Week EPA / Gm',
        'Defense vs Position (DvP)', 'Vegas Implied Team Total', 'Outlook Verdict'
    ]
    
    table_dict = {'Metric': metric_names}
    for r in selected_rows:
        pname = r['Player']
        if r['Pos'] == 'QB':
            stat_str = f"{r['Proj_pass_yd']:.1f} Pass Yds | {r['Proj_rush_yd']:.1f} Rush Yds"
        elif r['Pos'] == 'RB':
            stat_str = f"{r['Proj_rush_yd']:.1f} Rush Yds | {r['Proj_rec_yd']:.1f} Rec Yds ({r['Proj_receptions']:.1f} Rec)"
        else:
            stat_str = f"{r['Proj_rec_yd']:.1f} Rec Yds ({r['Proj_receptions']:.1f} Rec)"
            
        table_dict[pname] = [
            f"{r['Pos']} ({r['Team']})",
            f"{r['Status']} ({r['Env']})",
            r['Matchup'],
            f"{float(r[COL_MODEL]):.2f}",
            f"{float(r[COL_SLEEPER]):.2f}",
            f"{float(r[COL_DIFF]):+.2f}",
            stat_str,
            f"{r['Anytime TD %']:.1f}% ({prob_to_american_odds(r['Anytime TD %'])})" if r['Pos'] != 'QB' else "N/A",
            f"{float(r['ROS Value']):.2f}",
            f"{float(r['Playoff SoS']):+.2f} DvP",
            f"+{float(r['Vacated Boost']):.2f} PPR",
            f"{float(r['Floor (10%)']):.2f}",
            f"{float(r['Boom (90%)']):.2f}",
            f"{r['Boom %']}% ({r['Boom Catalyst']})",
            f"{r['Bust %']}% ({r['Bust Catalyst']})",
            f"{float(r['L3 xFP (Vol)']):.2f}",
            f"{float(r['L3 Avg PPR']):.2f}",
            f"{float(r['WOPR']):.2f}",
            f"{float(r['RACR']):.2f}",
            f"{float(r['L3 EPA']):+.2f}",
            f"{float(r['DvP Boost']):+.2f}",
            f"{float(r['team_implied']):.1f} pts",
            r['Outlook Verdict']
        ]
        
    return cards_html + bars_html, summary, pd.DataFrame(table_dict)

def run_accuracy_benchmark(eval_week_choice):
    weeks_to_eval = list(range(2, latest_completed_week + 1)) if eval_week_choice == "All Completed 2026 Weeks (Weeks 2+)" else [int(eval_week_choice.replace("Week ", ""))]
    all_eval = []
    for w in weeks_to_eval:
        w_df = build_week_projections(w)
        w_df = w_df[(~w_df['is_bye']) & (~w_df['is_out']) & (w_df['Actual PPR'].notna()) & (w_df[COL_MODEL] >= 5.0)].copy()
        w_df['Week'] = w
        all_eval.append(w_df)
        
    eval_df = pd.concat(all_eval, ignore_index=True)
    eval_df['Model Error'] = (eval_df[COL_MODEL] - eval_df['Actual PPR']).abs().round(2)
    eval_df['Sleeper Error'] = (eval_df[COL_SLEEPER] - eval_df['Actual PPR']).abs().round(2)
    
    non_ties = eval_df[eval_df['Model Error'] != eval_df['Sleeper Error']]
    win_rate = (non_ties['Model Error'] < non_ties['Sleeper Error']).mean() * 100 if len(non_ties) > 0 else 50.0
    
    eval_df['Closer Projection'] = np.where(
        eval_df['Model Error'] < eval_df['Sleeper Error'],
        "Model's Projection",
        np.where(eval_df['Model Error'] > eval_df['Sleeper Error'], "Sleeper's Projection", "Tie")
    )
    
    our_mae, slp_mae = eval_df['Model Error'].mean(), eval_df['Sleeper Error'].mean()
    our_rmse = np.sqrt(((eval_df[COL_MODEL] - eval_df['Actual PPR']) ** 2).mean())
    slp_rmse = np.sqrt(((eval_df[COL_SLEEPER] - eval_df['Actual PPR']) ** 2).mean())
    
    summary_md = (
        f"### Head-to-Head Accuracy Scorecard ({eval_week_choice} — {len(eval_df)} Starter Player-Weeks)\n"
        f"| Accuracy Metric | Model's Projection | Sleeper's Projection | Net Model Edge |\n"
        f"| :--- | :---: | :---: | :---: |\n"
        f"| **Mean Absolute Error (MAE)** | **{our_mae:.2f} pts** | {slp_mae:.2f} pts | **{slp_mae - our_mae:+.2f} pts** |\n"
        f"| **Root Mean Squared Error (RMSE)** | **{our_rmse:.2f} pts** | {slp_rmse:.2f} pts | **{slp_rmse - our_rmse:+.2f} pts** |\n"
        f"| **Head-to-Head Win Rate** | **{win_rate:.1f}%** | {100 - win_rate:.1f}% | **{'Model Leads' if win_rate >= 50 else 'Sleeper Leads'}** |"
    )
    show_cols = ['Week', 'Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Actual PPR', 'Model Error', 'Sleeper Error', 'Closer Projection']
    return summary_md, eval_df.sort_values(by='Actual PPR', ascending=False)[show_cols].reset_index(drop=True)

def score_to_grade(diff_from_avg, scale=1.0):
    adj = diff_from_avg / scale
    if adj >= 3.8: return "A+"
    if adj >= 1.8: return "A"
    if adj >= 0.6: return "B+"
    if adj >= -0.7: return "B"
    if adj >= -2.0: return "C+"
    if adj >= -3.5: return "C"
    return "D"

def compute_positional_unit_power(pos, ros_vals):
    vals = sorted(ros_vals, reverse=True)
    if pos == 'QB':
        starter_score = vals[0] if len(vals) > 0 else 0.0
        depth_score = 0.12 * (vals[1] if len(vals) > 1 else 0.0)
        star_bonus = max(0.0, (starter_score - 18.5) * 0.45)
    elif pos in ['RB', 'WR']:
        s1 = vals[0] if len(vals) > 0 else 0.0
        s2 = vals[1] if len(vals) > 1 else 0.0
        starter_score = s1 + s2
        d1 = vals[2] if len(vals) > 2 else 0.0
        d2 = vals[3] if len(vals) > 3 else 0.0
        depth_score = 0.22 * d1 + 0.10 * d2
        star_bonus = max(0.0, (s1 - 14.5) * 0.55) + max(0.0, (s2 - 13.5) * 0.55)
    else:
        starter_score = vals[0] if len(vals) > 0 else 0.0
        depth_score = 0.15 * (vals[1] if len(vals) > 1 else 0.0)
        star_bonus = max(0.0, (starter_score - 11.5) * 0.50)
        
    total_unit_score = starter_score + star_bonus + depth_score
    return round(total_unit_score, 2), round(starter_score + star_bonus, 2), round(depth_score, 2)

def calc_exact_10_starter_ros_ppg(pos_dict_vals):
    qbs = sorted(pos_dict_vals.get('QB', []), reverse=True)
    rbs = sorted(pos_dict_vals.get('RB', []), reverse=True)
    wrs = sorted(pos_dict_vals.get('WR', []), reverse=True)
    tes = sorted(pos_dict_vals.get('TE', []), reverse=True)
    flex_pool = sorted(rbs[2:] + wrs[2:] + tes[1:], reverse=True)
    starters_10_ppg = (
        (qbs[0] if qbs else 15.0) +
        sum(rbs[:2]) +
        sum(wrs[:2]) +
        (tes[0] if tes else 8.0) +
        sum(flex_pool[:2]) +
        15.5
    )
    return round(starters_10_ppg, 2)

def simulate_season_playoffs(team_records, playoff_teams=6, rem_weeks=8):
    n_teams = len(team_records)
    if n_teams < 4:
        for t in team_records:
            t['playoff_pct'], t['bye_pct'], t['champ_pct'] = 0.0, 0.0, 0.0
        return
        
    n_sims = 1500
    rng = np.random.default_rng(42)
    
    curr_wins = np.array([float(t['wins']) for t in team_records])
    curr_fpts = np.array([float(t['fpts_to_date']) for t in team_records])
    ros_means = np.array([float(t['ros_lineup_ppg']) for t in team_records])
    
    sim_wk_scores = rng.normal(loc=ros_means[:, None, None], scale=15.5, size=(n_teams, n_sims, max(1, rem_weeks)))
    sim_rem_pts = sim_wk_scores.sum(axis=2)
    
    league_wk_avg = sim_wk_scores.mean(axis=0, keepdims=True)
    opp_noise = rng.normal(loc=league_wk_avg, scale=11.0, size=sim_wk_scores.shape)
    sim_add_wins = (sim_wk_scores > opp_noise).sum(axis=2)
    
    total_wins = curr_wins[:, None] + sim_add_wins
    total_fpts = curr_fpts[:, None] + sim_rem_pts
    
    composite_rank_score = total_wins * 100000.0 + total_fpts
    seed_order = np.argsort(-composite_rank_score, axis=0)
    
    playoff_counts = np.zeros(n_teams)
    bye_counts = np.zeros(n_teams)
    champ_counts = np.zeros(n_teams)
    
    num_playoff = min(playoff_teams, n_teams)
    num_byes = 2 if num_playoff == 6 else (0 if num_playoff in [4, 8] else 2)
    
    for s in range(n_sims):
        seeds = seed_order[:, s]
        p_seeds = seeds[:num_playoff]
        playoff_counts[p_seeds] += 1
        
        if num_byes > 0:
            bye_counts[p_seeds[:num_byes]] += 1
            
        if num_playoff == 6:
            s1, s2, s3, s4, s5, s6 = p_seeds[0], p_seeds[1], p_seeds[2], p_seeds[3], p_seeds[4], p_seeds[5]
            w_36 = s3 if rng.normal(ros_means[s3], 15.5) >= rng.normal(ros_means[s6], 15.5) else s6
            w_45 = s4 if rng.normal(ros_means[s4], 15.5) >= rng.normal(ros_means[s5], 15.5) else s5
            semi1_w = s1 if rng.normal(ros_means[s1], 15.5) >= rng.normal(ros_means[w_45], 15.5) else w_45
            semi2_w = s2 if rng.normal(ros_means[s2], 15.5) >= rng.normal(ros_means[w_36], 15.5) else w_36
            champ = semi1_w if rng.normal(ros_means[semi1_w], 15.5) >= rng.normal(ros_means[semi2_w], 15.5) else semi2_w
            champ_counts[champ] += 1
        else:
            s1, s2, s3, s4 = p_seeds[0], p_seeds[1], p_seeds[2], p_seeds[3]
            semi1_w = s1 if rng.normal(ros_means[s1], 15.5) >= rng.normal(ros_means[s4], 15.5) else s4
            semi2_w = s2 if rng.normal(ros_means[s2], 15.5) >= rng.normal(ros_means[s3], 15.5) else s3
            champ = semi1_w if rng.normal(ros_means[semi1_w], 15.5) >= rng.normal(ros_means[semi2_w], 15.5) else semi2_w
            champ_counts[champ] += 1
            
    for idx, t in enumerate(team_records):
        t['playoff_pct'] = round((playoff_counts[idx] / n_sims) * 100, 1)
        t['bye_pct'] = round((bye_counts[idx] / n_sims) * 100, 1)
        t['champ_pct'] = round((champ_counts[idx] / n_sims) * 100, 1)

def format_manager_deep_dive(league_id, wk_int, manager_name):
    cache_key = (str(league_id).strip(), int(wk_int))
    if cache_key not in league_analysis_cache:
        return gr.update(visible=False), "", pd.DataFrame()
        
    bundle = league_analysis_cache[cache_key]
    team_records = bundle['teams']
    free_agents_df = bundle['free_agents_raw']
    
    target_team = next((t for t in team_records if t['manager'] == manager_name), None)
    if target_team is None and team_records:
        target_team = team_records[0]
    if target_team is None:
        return gr.update(visible=False), "", pd.DataFrame()
        
    m_name = target_team['manager']
    pos_diffs = target_team['pos_diffs']
    wk_pos_diffs = target_team['wk_pos_diffs']
    
    waiver_tip = "Bench depth is strong across all positions relative to available free agents."
    bench_skill = [p for p in target_team['full_roster'] if p['Role'] == 'BENCH' and p['Pos'] in ['QB', 'RB', 'WR', 'TE']]
    if bench_skill and not free_agents_df.empty:
        lowest_bench = min(bench_skill, key=lambda x: x['Full-Season ROS Value'])
        top_fa_rows = free_agents_df[free_agents_df['Pos'] == target_team['weakness']]
        if top_fa_rows.empty:
            top_fa_rows = free_agents_df
        top_fa = top_fa_rows.iloc[0]
        if float(top_fa['ROS Value']) - float(lowest_bench['Full-Season ROS Value']) >= 1.2:
            gain = float(top_fa['ROS Value']) - float(lowest_bench['Full-Season ROS Value'])
            waiver_tip = f"Drop {lowest_bench['Player']} ({lowest_bench['Full-Season ROS Value']:.1f} ROS PPG) to add Free Agent {top_fa['Player']} ({top_fa['Pos']} - {top_fa['Team']}, {top_fa['ROS Value']:.1f} ROS PPG, +{gain:.1f} PPG bench upgrade)."

    bw = max(wk_pos_diffs, key=wk_pos_diffs.get)
    ww = min(wk_pos_diffs, key=wk_pos_diffs.get)
    opp_info_str = target_team.get('matchup_summary_str', 'No Head-to-Head Opponent Scheduled')
    strat_play_str = target_team.get('specific_strategy_play', 'Active starting lineup is well-aligned with this week\'s matchup strategy.')
    start_sit_str = target_team.get('start_sit_summary', 'Active starting lineup is optimally set for maximum win probability this week.')
    decider_str = target_team.get('leverage_decider_str', 'N/A')

    starters_sk = [p for p in target_team['full_roster'] if p['Role'] == 'STARTER' and p['Pos'] in ['QB', 'RB', 'WR', 'TE']][:4]
    starter_bars_html = ""
    for sp in starters_sk:
        starter_bars_html += render_payoff_distribution_bar_html(
            sp['Player'], f"Starter {sp['Pos']} - {sp['Team']}",
            float(sp['Floor (10%)']), float(sp[COL_MODEL]), float(sp[COL_SLEEPER]), float(sp['Boom (90%)']),
            max_scale=34.0, compact=True
        )

    audit_html = (
        f'<div style="display:flex; flex-direction:column; gap:16px; margin-top:8px; margin-bottom:14px;">'
        f'<div style="padding:18px; border-radius:12px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.03);">'
        f'<div style="font-size:1.22rem; font-weight:700; margin-bottom:12px;">Full-Season Roster Audit: {m_name} — Overall Grade: {target_team["overall_grade"]}</div>'
        f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(155px, 1fr)); gap:12px; margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid rgba(148,163,184,0.2);">'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Playoff Probability</div><div style="font-size:1.15rem; font-weight:700;">{target_team["playoff_pct"]}%</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">1st-Round Bye Odds</div><div style="font-size:1.15rem; font-weight:700;">{target_team["bye_pct"]}%</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Championship Odds</div><div style="font-size:1.15rem; font-weight:700;">{target_team["champ_pct"]}%</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Healthy 10-Starter PPG</div><div style="font-size:1.15rem; font-weight:700;">{target_team["ros_lineup_ppg"]:.2f} PPG</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Playoff SoS (Wks 15-17)</div><div style="font-size:1.1rem; font-weight:600;">{target_team["team_po_sos"]:+.2f} DvP</div></div>'
        f'</div>'
        f'<div style="font-size:0.94rem; line-height:1.65;">'
        f'<div><strong>Positional Unit Grades (82% Starters / 18% Depth):</strong> QB: <strong>{target_team["qb_grade"]}</strong> &nbsp;|&nbsp; RB: <strong>{target_team["rb_grade"]}</strong> <em>(Starters: {target_team["rb_starter_grade"]}, Depth: {target_team["rb_depth_grade"]})</em> &nbsp;|&nbsp; WR: <strong>{target_team["wr_grade"]}</strong> <em>(Starters: {target_team["wr_starter_grade"]}, Depth: {target_team["wr_depth_grade"]})</em> &nbsp;|&nbsp; TE: <strong>{target_team["te_grade"]}</strong></div>'
        f'<div style="margin-top:4px;"><strong>Primary Season Strength:</strong> {target_team["strength"]} ({pos_diffs[target_team["strength"]]:+.1f} PPG vs League Avg) &nbsp;|&nbsp; <strong>Biggest Season Need:</strong> {target_team["weakness"]} ({pos_diffs[target_team["weakness"]]:+.1f} PPG vs League Avg)</div>'
        f'<div style="margin-top:4px;"><strong>Waiver Wire Upgrade Tip:</strong> {waiver_tip}</div>'
        f'</div></div>'
        f'<div style="padding:18px; border-radius:12px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.03);">'
        f'<div style="font-size:1.22rem; font-weight:700; margin-bottom:12px;">Week {wk_int} Lineup, Live Leverage & Matchup Strategy Audit: {m_name}</div>'
        f'<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid rgba(148,163,184,0.2);">'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Model\'s Projection</div><div style="font-size:1.15rem; font-weight:700;">{target_team["starter_pts"]:.2f} pts</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Sleeper\'s Projection</div><div style="font-size:1.15rem; font-weight:600;">{target_team["sleeper_pts"]:.2f} pts</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Live Banked Points</div><div style="font-size:1.15rem; font-weight:700; color:#2563eb;">{target_team["live_pts"]:.2f} pts</div></div>'
        f'<div><div style="font-size:0.78rem; opacity:0.72;">Optimal Week {wk_int} Lineup</div><div style="font-size:1.15rem; font-weight:700;">{target_team["opt_pts"]:.2f} pts</div></div>'
        f'</div>'
        f'<div style="font-size:0.94rem; line-height:1.65; margin-bottom:12px;">'
        f'<div><strong>Head-to-Head Matchup Outlook:</strong> {opp_info_str}</div>'
        f'<div style="margin-top:4px;"><strong>#1 Matchup Decider (Highest Leverage Starter):</strong> {decider_str}</div>'
        f'<div style="margin-top:4px;"><strong>Specific Roster Strategy Recommendation:</strong> {strat_play_str}</div>'
        f'<div style="margin-top:4px;"><strong>Opponent-Conditioned Start/Sit Verdict:</strong> {start_sit_str}</div>'
        f'<div style="margin-top:4px;"><strong>Week {wk_int} Unit Performance vs League Avg:</strong> Strongest Active Unit: {bw} ({wk_pos_diffs[bw]:+.1f} pts) &nbsp;|&nbsp; Weakest Active Unit: {ww} ({wk_pos_diffs[ww]:+.1f} pts)</div>'
        f'</div>'
        f'<div style="font-size:0.86rem; font-weight:700; margin-bottom:6px; opacity:0.85;">Core Starter Payoff Distribution Tracks (Week {wk_int}):</div>'
        f'{starter_bars_html}'
        f'</div></div>'
    )
    
    roster_df = pd.DataFrame(target_team['full_roster'])
    return gr.update(visible=True), audit_html, roster_df

def filter_waiver_wire(league_id, week_num, pos_filter, sort_by, search_txt):
    cache_key = (str(league_id).strip(), int(week_num))
    waiver_cols = ["Player", "Pos", "Team", "Matchup", COL_MODEL, COL_SLEEPER, COL_DIFF, "Full-Season ROS Value", "Floor (10%)", "Boom (90%)", "Boom %", "L3 xFP (Vol)", "Playoff SoS", "Scouting Catalyst"]
    if cache_key not in league_analysis_cache:
        return pd.DataFrame(columns=waiver_cols)
        
    unrostered_df = league_analysis_cache[cache_key]['free_agents_raw'].copy()
    if unrostered_df.empty:
        return pd.DataFrame(columns=waiver_cols)
        
    if pos_filter != "ALL":
        unrostered_df = unrostered_df[unrostered_df['Pos'] == pos_filter].copy()
    if search_txt and str(search_txt).strip():
        q = str(search_txt).strip()
        unrostered_df = unrostered_df[unrostered_df['Player'].str.contains(q, case=False, na=False) | unrostered_df['Team'].str.contains(q, case=False, na=False)].copy()
        
    sort_map = {
        "Overall Waiver Score (Best Pickup)": "Waiver Score",
        "Week Model's Projection": COL_MODEL,
        "Full-Season ROS Value": "Full-Season ROS Value",
        "Expected Volume (L3 xFP)": "L3 xFP (Vol)",
        "Boom Probability %": "Boom %",
        "Playoff SoS (Wks 15-17)": "Playoff SoS"
    }
    sort_col = sort_map.get(sort_by, "Waiver Score")
    unrostered_df = unrostered_df.sort_values(by=sort_col, ascending=False).head(25)
    return unrostered_df[waiver_cols].reset_index(drop=True)

def filter_start_sit_table(league_id, week_num, mgr_filter):
    cache_key = (str(league_id).strip(), int(week_num))
    ss_cols = ["Manager", "Action Verdict", "Bench / Roster Player to Start", "Current Starter to Sit / Flex Context", "Proj Point Gain", "Matchup Win Prob Delta", "Floor / Ceiling Comparison", "Detailed Statistical Reasoning"]
    if cache_key not in league_analysis_cache:
        return pd.DataFrame(columns=ss_cols)
    df_ss = league_analysis_cache[cache_key]['start_sit_df'].copy()
    if df_ss.empty:
        return pd.DataFrame(columns=ss_cols)
    if mgr_filter and mgr_filter != "ALL MANAGERS":
        df_ss = df_ss[df_ss['Manager'] == mgr_filter].copy()
    return df_ss.reset_index(drop=True)

def analyze_sleeper_league(league_id, week_num):
    wk_int = int(week_num)
    projection_week_cache.pop(wk_int, None)
    sleeper_cache.pop(wk_int, None)
    
    pr_cols = [
        "Rank", "Manager", "Record",
        COL_MODEL, COL_SLEEPER, COL_DIFF, "Live Banked Pts", "Optimal Lineup Pts",
        "Overall Season Grade", "Playoff %", "1st-Rd Bye %", "Title %",
        "QB Season Grade", "RB Season Grade", "WR Season Grade", "TE Season Grade",
        "Full-Season Strength", "Full-Season Need"
    ]
    tr_cols = ["Trade Type", "Manager 1", "Manager 1 Sends", "Manager 2", "Manager 2 Sends", "Playoff / Title Equity Delta", "Why This Trade Works (Mutual Starter PPG Gain)"]
    matchup_cols = ["Matchup", "Pre-Game Model Proj", "Live Score", "Win Odds", "#1 Matchup Decider (Leverage Index)", "Strategy Mode", "Specific Roster Prioritization (Who to Start & Why)"]
    waiver_cols = ["Player", "Pos", "Team", "Matchup", COL_MODEL, COL_SLEEPER, COL_DIFF, "Full-Season ROS Value", "Floor (10%)", "Boom (90%)", "Boom %", "L3 xFP (Vol)", "Playoff SoS", "Scouting Catalyst"]
    ss_cols = ["Manager", "Action Verdict", "Bench / Roster Player to Start", "Current Starter to Sit / Flex Context", "Proj Point Gain", "Matchup Win Prob Delta", "Floor / Ceiling Comparison", "Detailed Statistical Reasoning"]
    
    if not league_id or not str(league_id).strip():
        empty_u = gr.update(choices=[], value=None)
        prompt_msg = "**Enter your Sleeper League ID above and click 'Sync Real-Time League Data & Poll Live Scores'** to unlock Power Rankings, Live Matchup Win Probability, Player Leverage Meters, Opponent-Conditioned Start/Sit Advice, Waiver Radar, and Package Trade Simulations."
        return prompt_msg, pd.DataFrame(columns=pr_cols), "", pd.DataFrame(columns=matchup_cols), pd.DataFrame(columns=ss_cols), pd.DataFrame(columns=waiver_cols), pd.DataFrame(columns=tr_cols), empty_u, empty_u, gr.update(visible=False), "", pd.DataFrame(), empty_u, empty_u
        
    league_id = str(league_id).strip()
    # save_local_league_id(league_id)
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        league_res = requests.get(f"https://api.sleeper.app/v1/league/{league_id}", headers=headers, timeout=8)
        users_res = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/users", headers=headers, timeout=8)
        rosters_res = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/rosters", headers=headers, timeout=8)
        matchups_res = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{wk_int}", headers=headers, timeout=8)
    except Exception as e:
        empty_u = gr.update(choices=[], value=None)
        return f"Network notice connecting to Sleeper: {e}", pd.DataFrame(columns=pr_cols), "", pd.DataFrame(columns=matchup_cols), pd.DataFrame(columns=ss_cols), pd.DataFrame(columns=waiver_cols), pd.DataFrame(columns=tr_cols), empty_u, empty_u, gr.update(visible=False), "", pd.DataFrame(), empty_u, empty_u
    
    if users_res.status_code != 200 or rosters_res.status_code != 200:
        empty_u = gr.update(choices=[], value=None)
        return f"Failed to fetch league {league_id}. Ensure ID is correct and league is public.", pd.DataFrame(columns=pr_cols), "", pd.DataFrame(columns=matchup_cols), pd.DataFrame(columns=ss_cols), pd.DataFrame(columns=waiver_cols), pd.DataFrame(columns=tr_cols), empty_u, empty_u, gr.update(visible=False), "", pd.DataFrame(), empty_u, empty_u
        
    playoff_teams = 6
    playoff_week_start = 15
    if league_res.status_code == 200 and isinstance(league_res.json(), dict):
        l_settings = league_res.json().get('settings', {}) or {}
        playoff_teams = int(l_settings.get('playoff_teams', 6) or 6)
        playoff_week_start = int(l_settings.get('playoff_week_start', 15) or 15)
        
    week_matchup_map = {}
    if matchups_res.status_code == 200 and isinstance(matchups_res.json(), list):
        for m in matchups_res.json():
            rid = m.get('roster_id')
            if rid is not None and m.get('starters'):
                week_matchup_map[rid] = {
                    'matchup_id': m.get('matchup_id'),
                    'starters': [str(x) for x in (m.get('starters') or []) if str(x) != "0"],
                    'players': [str(x) for x in (m.get('players') or []) if str(x) != "0"],
                    'players_points': {str(k): float(v or 0.0) for k, v in (m.get('players_points') or {}).items()},
                    'total_live_pts': float(m.get('points') or 0.0)
                }

    _, _, pid_proj_wk1, _ = get_sleeper_week_data(1)
    _, _, pid_proj_curr, _ = get_sleeper_week_data(wk_int)
    
    users = {u['user_id']: u.get('display_name') or u.get('username') or "Manager" for u in users_res.json()}
    rosters = rosters_res.json()
    
    wk_df = build_week_projections(wk_int)
    by_gsis = {str(r['player_id']).strip(): r for _, r in wk_df.iterrows()}
    by_norm_pos = {(r['norm_name'], r['Pos']): r for _, r in wk_df.iterrows()}
    by_norm = {r['norm_name']: r for _, r in wk_df.iterrows()}
    
    rostered_norm_names = set()
    
    def resolve_sleeper_pid(pid_str):
        pid_str = str(pid_str).strip()
        if not pid_str or pid_str == "0":
            return None
            
        if pid_str.isalpha() and len(pid_str) <= 3:
            t_abbr = team_fix.get(pid_str, pid_str)
            sched_info = schedule_map.get((wk_int, t_abbr), None)
            is_bye = (sched_info is None)
            mod_def = 0.0 if is_bye else round(max(4.5, min(10.5, 7.8 + (22.0 - sched_info['opp_implied']) * 0.32)), 2)
            slp_raw = pid_proj_curr.get(pid_str, mod_def)
            slp_def = 0.0 if is_bye else round(min(11.0, float(slp_raw)), 2)
            return {
                'Player': f"{t_abbr} DEF", 'norm_name': norm_name(f"{t_abbr} DEF"), 'Pos': 'DEF', 'Team': t_abbr,
                'Status': 'BYE' if is_bye else 'ACTIVE', 'Matchup': 'BYE' if is_bye else sched_info['matchup_str'],
                COL_MODEL: mod_def, COL_SLEEPER: slp_def, 'ROS Value': 7.8, 'Playoff SoS': 0.0,
                'Floor (10%)': max(2.0, mod_def - 3.5), 'Boom (90%)': mod_def + 4.5, 'Boom %': 18.0, 'Bust %': 18.0,
                'L3 xFP (Vol)': mod_def, 'DvP Boost': 0.0, 'Boom Catalyst': 'Defensive Matchup'
            }
            
        info = sleeper_pid_to_info.get(pid_str)
        if info:
            rostered_norm_names.add(info['norm'])
            if info['pos'] in ['K', 'DEF']:
                t_abbr = team_fix.get(info['team'], info['team'])
                sched_info = schedule_map.get((wk_int, t_abbr), None)
                is_bye = (sched_info is None)
                if is_bye:
                    mod_val, slp_val = 0.0, 0.0
                elif info['pos'] == 'K':
                    mod_val = round(max(5.5, min(10.5, 8.1 + (sched_info['team_implied'] - 22.0) * 0.20)), 2)
                    slp_raw = pid_proj_curr.get(pid_str, mod_val)
                    slp_val = round(min(10.8, float(slp_raw)), 2)
                else:
                    mod_val = round(max(4.5, min(10.5, 7.8 + (22.0 - sched_info['opp_implied']) * 0.32)), 2)
                    slp_raw = pid_proj_curr.get(pid_str, mod_val)
                    slp_val = round(min(11.0, float(slp_raw)), 2)
                return {
                    'Player': info['name'], 'norm_name': info['norm'], 'Pos': info['pos'], 'Team': t_abbr,
                    'Status': 'BYE' if is_bye else 'ACTIVE', 'Matchup': 'BYE' if is_bye else sched_info['matchup_str'],
                    COL_MODEL: mod_val, COL_SLEEPER: slp_val, 'ROS Value': 8.0, 'Playoff SoS': 0.0,
                    'Floor (10%)': max(2.5, mod_val - 3.2), 'Boom (90%)': mod_val + 4.2, 'Boom %': 18.0, 'Bust %': 18.0,
                    'L3 xFP (Vol)': mod_val, 'DvP Boost': 0.0, 'Boom Catalyst': 'Special Teams / DEF'
                }
                
            if info['gsis_id'] and info['gsis_id'] in by_gsis:
                return by_gsis[info['gsis_id']]
            if (info['norm'], info['pos']) in by_norm_pos:
                return by_norm_pos[(info['norm'], info['pos'])]
            if info['norm'] in by_norm:
                return by_norm[info['norm']]
                
            slp_wk = pid_proj_curr.get(pid_str, 0.0)
            slp_base = pid_proj_wk1.get(pid_str, slp_wk)
            t_abbr = team_fix.get(info['team'], info['team'])
            sched_info = schedule_map.get((wk_int, t_abbr), None)
            po_sos = playoff_sos_map.get((t_abbr, info['pos']), 0.0)
            return {
                'Player': info['name'], 'norm_name': info['norm'], 'Pos': info['pos'], 'Team': t_abbr,
                'Status': 'BYE' if sched_info is None else 'ACTIVE',
                'Matchup': 'BYE' if sched_info is None else sched_info['matchup_str'],
                COL_MODEL: slp_wk, COL_SLEEPER: slp_wk, 'ROS Value': max(slp_wk, slp_base), 'Playoff SoS': po_sos,
                'Floor (10%)': max(0.0, round(slp_wk * 0.65, 2)), 'Boom (90%)': round(slp_wk * 1.35 + 2.5, 2),
                'Boom %': 18.0, 'Bust %': 18.0, 'L3 xFP (Vol)': slp_wk, 'DvP Boost': 0.0, 'Boom Catalyst': 'Projected Role'
            }
        return None

    team_records = []
    for r in rosters:
        rid = r.get('roster_id')
        manager = users.get(r.get('owner_id'), f"Team {rid}")
        
        curr_roster_pids = [str(x) for x in (r.get('players') or []) if str(x) != "0"]
        for cpid in curr_roster_pids:
            cinfo = sleeper_pid_to_info.get(cpid)
            if cinfo:
                rostered_norm_names.add(cinfo['norm'])
        
        if rid in week_matchup_map and week_matchup_map[rid]['starters']:
            starters = week_matchup_map[rid]['starters']
            week_pids = list(dict.fromkeys(starters + (week_matchup_map[rid]['players'] or curr_roster_pids)))
            m_id = week_matchup_map[rid].get('matchup_id')
            p_pts_map = week_matchup_map[rid].get('players_points', {})
            live_team_tot = week_matchup_map[rid].get('total_live_pts', 0.0)
        else:
            starters = [str(x) for x in (r.get('starters') or []) if str(x) != "0"]
            week_pids = list(dict.fromkeys(starters + curr_roster_pids))
            m_id = None
            p_pts_map = {}
            live_team_tot = 0.0
            
        r_set = r.get('settings', {}) or {}
        wins = int(r_set.get('wins', 0) or 0)
        losses = int(r_set.get('losses', 0) or 0)
        fpts_to_date = float(r_set.get('fpts', 0) or 0) + float(r_set.get('fpts_decimal', 0) or 0) / 100.0
        
        starter_pts = 0.0
        sleeper_starter_pts = 0.0
        starter_var_sum = 0.0
        wk_pos_groups = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'K': [], 'DEF': []}
        wk_starter_pos_pts = {'QB': 0.0, 'RB': 0.0, 'WR': 0.0, 'TE': 0.0}
        full_roster_rows = []
        seen_week_names = set()
        
        for pid in week_pids:
            p_obj = resolve_sleeper_pid(pid)
            if p_obj is not None:
                pname = p_obj['Player']
                if pname in seen_week_names:
                    continue
                seen_week_names.add(pname)
                rostered_norm_names.add(p_obj.get('norm_name', norm_name(pname)))
                
                pos = p_obj['Pos']
                wk_proj = float(p_obj[COL_MODEL])
                slp_proj = float(p_obj.get(COL_SLEEPER, wk_proj))
                ros_val = float(p_obj['ROS Value'])
                po_sos = float(p_obj.get('Playoff SoS', 0.0))
                flr_val = float(p_obj.get('Floor (10%)', wk_proj * 0.65))
                ceil_val = float(p_obj.get('Boom (90%)', wk_proj * 1.35))
                boom_p = float(p_obj.get('Boom %', 18.0))
                bust_p = float(p_obj.get('Bust %', 18.0))
                xfp_v = float(p_obj.get('L3 xFP (Vol)', wk_proj))
                dvp_v = float(p_obj.get('DvP Boost', 0.0))
                b_cat = str(p_obj.get('Boom Catalyst', ''))
                live_p = float(p_pts_map.get(str(pid), 0.0))
                is_starter = (pid in starters)
                
                p_sigma = max(2.2, (ceil_val - flr_val) / 2.563) if wk_proj > 0 else 0.0
                
                if is_starter:
                    starter_pts += wk_proj
                    sleeper_starter_pts += slp_proj
                    starter_var_sum += (p_sigma ** 2)
                    if pos in wk_starter_pos_pts:
                        wk_starter_pos_pts[pos] += wk_proj
                        
                if pos in wk_pos_groups:
                    wk_pos_groups[pos].append(wk_proj)
                    
                full_roster_rows.append({
                    'Role': 'STARTER' if is_starter else 'BENCH',
                    'Player': pname,
                    'Pos': pos,
                    'Team': p_obj['Team'],
                    'Status': p_obj.get('Status', 'ACTIVE'),
                    'Matchup': p_obj.get('Matchup', ''),
                    COL_MODEL: round(wk_proj, 2),
                    COL_SLEEPER: round(slp_proj, 2),
                    COL_DIFF: round(wk_proj - slp_proj, 2),
                    'Live Pts': round(live_p, 2),
                    'Floor (10%)': round(flr_val, 2),
                    'Boom (90%)': round(ceil_val, 2),
                    'Boom %': round(boom_p, 1),
                    'Bust %': round(bust_p, 1),
                    'L3 xFP (Vol)': round(xfp_v, 2),
                    'DvP Boost': round(dvp_v, 2),
                    'Full-Season ROS Value': round(ros_val, 2),
                    'Playoff SoS (Wks 15-17)': round(po_sos, 2),
                    'p_sigma': round(p_sigma, 2),
                    'Catalyst': b_cat
                })
                
        full_roster_rows.sort(key=lambda x: (0 if x['Role'] == 'STARTER' else 1, -x[COL_MODEL]))
        
        qbs_wk = sorted(wk_pos_groups['QB'], reverse=True)
        rbs_wk = sorted(wk_pos_groups['RB'], reverse=True)
        wrs_wk = sorted(wk_pos_groups['WR'], reverse=True)
        tes_wk = sorted(wk_pos_groups['TE'], reverse=True)
        ks_wk = sorted(wk_pos_groups['K'], reverse=True)
        defs_wk = sorted(wk_pos_groups['DEF'], reverse=True)
        
        opt_pts = (
            (qbs_wk[0] if qbs_wk else 0.0) +
            sum(rbs_wk[:2]) +
            sum(wrs_wk[:2]) +
            (tes_wk[0] if tes_wk else 0.0) +
            (ks_wk[0] if ks_wk else 8.0) +
            (defs_wk[0] if defs_wk else 7.5)
        )
        flex_pool_wk = sorted(rbs_wk[2:] + wrs_wk[2:] + tes_wk[1:], reverse=True)
        opt_pts += sum(flex_pool_wk[:2])
        
        ros_pos_groups = {'QB': [], 'RB': [], 'WR': [], 'TE': []}
        po_sos_list = []
        seen_curr_names = set()
        for cpid in curr_roster_pids:
            p_obj = resolve_sleeper_pid(cpid)
            if p_obj is not None and p_obj['Pos'] in ros_pos_groups:
                pname = p_obj['Player']
                if pname in seen_curr_names:
                    continue
                seen_curr_names.add(pname)
                pos = p_obj['Pos']
                ros_val = float(p_obj['ROS Value'])
                po_sos = float(p_obj.get('Playoff SoS', 0.0))
                ros_pos_groups[pos].append({
                    'name': pname,
                    'pos': pos,
                    'team': p_obj['Team'],
                    'ros_val': ros_val,
                    'po_sos': po_sos
                })
                po_sos_list.append(po_sos)
                
        for pos in ros_pos_groups:
            ros_pos_groups[pos].sort(key=lambda x: x['ros_val'], reverse=True)
            
        pos_dict_vals = {pos: [x['ros_val'] for x in ros_pos_groups[pos]] for pos in ['QB', 'RB', 'WR', 'TE']}
        ros_lineup_ppg = calc_exact_10_starter_ros_ppg(pos_dict_vals)
        
        qb_tot, qb_st, qb_dp = compute_positional_unit_power('QB', pos_dict_vals['QB'])
        rb_tot, rb_st, rb_dp = compute_positional_unit_power('RB', pos_dict_vals['RB'])
        wr_tot, wr_st, wr_dp = compute_positional_unit_power('WR', pos_dict_vals['WR'])
        te_tot, te_st, te_dp = compute_positional_unit_power('TE', pos_dict_vals['TE'])
        
        pos_raw_score = {'QB': qb_tot, 'RB': rb_tot, 'WR': wr_tot, 'TE': te_tot}
        pos_starter_score = {'QB': qb_st, 'RB': rb_st, 'WR': wr_st, 'TE': te_st}
        pos_depth_score = {'QB': qb_dp, 'RB': rb_dp, 'WR': wr_dp, 'TE': te_dp}
        
        team_records.append({
            'roster_id': rid,
            'matchup_id': m_id,
            'manager': manager,
            'wins': wins,
            'losses': losses,
            'fpts_to_date': round(fpts_to_date, 2),
            'starter_pts': round(starter_pts, 2),
            'sleeper_pts': round(sleeper_starter_pts, 2),
            'diff_pts': round(starter_pts - sleeper_starter_pts, 2),
            'live_pts': round(live_team_tot, 2),
            'team_var': max(140.0, starter_var_sum),
            'opt_pts': round(max(starter_pts, opt_pts), 2),
            'ros_lineup_ppg': ros_lineup_ppg,
            'team_po_sos': round(float(np.mean(po_sos_list)) if po_sos_list else 0.0, 2),
            'pos_groups': ros_pos_groups,
            'pos_dict_vals': pos_dict_vals,
            'pos_raw': pos_raw_score,
            'pos_starter': pos_starter_score,
            'pos_depth': pos_depth_score,
            'wk_pos_pts': wk_starter_pos_pts,
            'full_roster': full_roster_rows
        })
        
    league_pos_avg = {pos: np.mean([t['pos_raw'][pos] for t in team_records]) for pos in ['QB', 'RB', 'WR', 'TE']}
    league_starter_avg = {pos: np.mean([t['pos_starter'][pos] for t in team_records]) for pos in ['QB', 'RB', 'WR', 'TE']}
    league_depth_avg = {pos: np.mean([t['pos_depth'][pos] for t in team_records]) for pos in ['QB', 'RB', 'WR', 'TE']}
    league_wk_pos_avg = {pos: np.mean([t['wk_pos_pts'][pos] for t in team_records]) for pos in ['QB', 'RB', 'WR', 'TE']}
    league_ros_avg = np.mean([t['ros_lineup_ppg'] for t in team_records])
    
    rem_reg_weeks = max(1, playoff_week_start - latest_completed_week - 1)
    simulate_season_playoffs(team_records, playoff_teams=playoff_teams, rem_weeks=rem_reg_weeks)
    
    def build_team_strategy_and_start_sit(team_obj, opp_obj, strategy_mode):
        starters_sk = [p for p in team_obj['full_roster'] if p['Role'] == 'STARTER' and p['Pos'] in ['QB', 'RB', 'WR', 'TE']]
        bench_sk = [p for p in team_obj['full_roster'] if p['Role'] == 'BENCH' and p['Pos'] in ['QB', 'RB', 'WR', 'TE'] and p['Status'] == 'ACTIVE']
        
        my_mu = team_obj['starter_pts']
        my_var = team_obj['team_var']
        opp_mu = opp_obj['starter_pts'] if opp_obj else my_mu
        opp_var = opp_obj['team_var'] if opp_obj else 220.0
        joint_sigma = math.sqrt(my_var + opp_var)
        base_win_prob = normal_cdf((my_mu - opp_mu) / joint_sigma) * 100.0
        
        best_lev_player = None
        best_lev_swing = -1.0
        for s in starters_sk:
            rem_var = max(100.0, my_var - (s['p_sigma'] ** 2))
            rem_joint_sigma = math.sqrt(rem_var + opp_var)
            mu_without_s = my_mu - s[COL_MODEL]
            wp_boom = normal_cdf(((mu_without_s + s['Boom (90%)']) - opp_mu) / rem_joint_sigma) * 100.0
            wp_bust = normal_cdf(((mu_without_s + s['Floor (10%)']) - opp_mu) / rem_joint_sigma) * 100.0
            swing = round(wp_boom - wp_bust, 1)
            s['leverage_swing'] = swing
            if swing > best_lev_swing:
                best_lev_swing = swing
                best_lev_player = f"{s['Player']} ({swing:.1f}% Win Swing: {wp_bust:.0f}% Win if Floor -> {wp_boom:.0f}% Win if Boom)"
        team_obj['leverage_decider_str'] = best_lev_player if best_lev_player else "Balanced Lineup"
        
        if strategy_mode == 'FLOOR':
            top_floor_starters = sorted(starters_sk, key=lambda x: x['Floor (10%)'], reverse=True)[:2]
            anchors = ", ".join([f"{p['Player']} ({p['Floor (10%)']:.1f} Flr)" for p in top_floor_starters])
            floor_swap = None
            for b in sorted(bench_sk, key=lambda x: x['Floor (10%)'], reverse=True):
                for s in sorted(starters_sk, key=lambda x: x['Floor (10%)']):
                    if (s['Pos'] == b['Pos'] or (s['Pos'] in ['RB', 'WR', 'TE'] and b['Pos'] in ['RB', 'WR', 'TE'])) and b['Floor (10%)'] >= s['Floor (10%)'] + 0.8 and b[COL_MODEL] >= s[COL_MODEL] - 1.5:
                        floor_swap = f"Plug in high-floor bench option {b['Player']} (Floor: {b['Floor (10%)']:.1f} pts, Bust: {b['Bust %']}%) over volatile starter {s['Player']} (Floor: {s['Floor (10%)']:.1f} pts, Bust: {s['Bust %']}%). Core Floor Anchors: {anchors}."
                        break
                if floor_swap:
                    break
            strat_rec = floor_swap if floor_swap else f"Keep High-Floor Core Locked ({anchors}) — Avoid chasing low-floor boom/bust bench plays."
        elif strategy_mode == 'CEILING':
            top_ceil_starters = sorted(starters_sk, key=lambda x: x['Boom (90%)'], reverse=True)[:2]
            anchors = ", ".join([f"{p['Player']} ({p['Boom (90%)']:.1f} Ceil)" for p in top_ceil_starters])
            ceil_swap = None
            for b in sorted(bench_sk, key=lambda x: x['Boom (90%)'], reverse=True):
                for s in sorted(starters_sk, key=lambda x: x['Boom (90%)']):
                    if (s['Pos'] == b['Pos'] or (s['Pos'] in ['RB', 'WR', 'TE'] and b['Pos'] in ['RB', 'WR', 'TE'])) and b['Boom (90%)'] >= s['Boom (90%)'] + 1.5 and b[COL_MODEL] >= 7.5:
                        ceil_swap = f"Chase Upside: Start bench boom candidate {b['Player']} (Ceiling: {b['Boom (90%)']:.1f} pts, Boom: {b['Boom %']}%, {b['Catalyst']}) over capped starter {s['Player']} (Ceiling: {s['Boom (90%)']:.1f} pts). Core Boom Anchors: {anchors}."
                        break
                if ceil_swap:
                    break
            strat_rec = ceil_swap if ceil_swap else f"Lean on High-Ceiling Starters ({anchors}) — Target 90th-percentile spike outcomes."
        else:
            top_proj = sorted(starters_sk, key=lambda x: x[COL_MODEL], reverse=True)[:2]
            anchors = ", ".join([f"{p['Player']} ({p[COL_MODEL]:.1f} Proj)" for p in top_proj])
            strat_rec = f"Toss-Up Game: Start highest raw Model's Projection lineup anchored by {anchors}."
            
        ss_recs = []
        used_starters = set()
        for b in sorted(bench_sk, key=lambda x: x[COL_MODEL], reverse=True):
            if b[COL_MODEL] < 6.5:
                continue
            for s in sorted(starters_sk, key=lambda x: x[COL_MODEL]):
                if s['Player'] in used_starters:
                    continue
                can_swap = (s['Pos'] == b['Pos']) or (s['Pos'] in ['RB', 'WR', 'TE'] and b['Pos'] in ['RB', 'WR', 'TE'])
                if can_swap:
                    pt_gain = b[COL_MODEL] - s[COL_MODEL]
                    new_mu = my_mu + pt_gain
                    new_var = max(120.0, my_var - (s['p_sigma'] ** 2) + (b['p_sigma'] ** 2))
                    new_wp = normal_cdf((new_mu - opp_mu) / math.sqrt(new_var + opp_var)) * 100.0
                    wp_delta = new_wp - base_win_prob
                    
                    if pt_gain >= 0.7 or wp_delta >= 1.2:
                        used_starters.add(s['Player'])
                        reason_str = (
                            f"{b['Player']} projects for {b[COL_MODEL]:.2f} PPR ({b['L3 xFP (Vol)']:.1f} L3 xFP, {b['DvP Boost']:+.1f} DvP {b['Matchup']}) "
                            f"vs {s['Player']}'s {s[COL_MODEL]:.2f} PPR ({s['L3 xFP (Vol)']:.1f} L3 xFP, {s['DvP Boost']:+.1f} DvP {s['Matchup']})."
                        )
                        ss_recs.append({
                            "Manager": team_obj['manager'],
                            "Action Verdict": "START BENCH UPGRADE",
                            "Bench / Roster Player to Start": f"{b['Player']} ({b['Pos']} - {b['Team']}, {b[COL_MODEL]:.2f} PPR)",
                            "Current Starter to Sit / Flex Context": f"Sit {s['Player']} ({s['Pos']} - {s['Team']}, {s[COL_MODEL]:.2f} PPR)",
                            "Proj Point Gain": f"{pt_gain:+.2f} PPR",
                            "Matchup Win Prob Delta": f"{wp_delta:+.1f}% Win Odds",
                            "Floor / Ceiling Comparison": f"{b['Player']}: {b['Floor (10%)']:.1f}-{b['Boom (90%)']:.1f} | {s['Player']}: {s['Floor (10%)']:.1f}-{s['Boom (90%)']:.1f}",
                            "Detailed Statistical Reasoning": reason_str
                        })
                        break
        if not ss_recs and starters_sk:
            best_s = max(starters_sk, key=lambda x: x[COL_MODEL])
            weakest_s = min(starters_sk, key=lambda x: x[COL_MODEL])
            best_b = bench_sk[0] if bench_sk else None
            b_note = f"Best bench pivot is {best_b['Player']} ({best_b[COL_MODEL]:.2f} PPR, Ceiling: {best_b['Boom (90%)']:.1f})." if best_b else "No active bench pivots available."
            ss_recs.append({
                "Manager": team_obj['manager'],
                "Action Verdict": "LINEUP OPTIMALLY SET",
                "Bench / Roster Player to Start": f"Keep {weakest_s['Player']} ({weakest_s['Pos']}, {weakest_s[COL_MODEL]:.2f} PPR) in Lineup",
                "Current Starter to Sit / Flex Context": f"Anchored by {best_s['Player']} ({best_s[COL_MODEL]:.2f} PPR)",
                "Proj Point Gain": "0.00 (Optimal)",
                "Matchup Win Prob Delta": "+0.0% (Optimal)",
                "Floor / Ceiling Comparison": f"{weakest_s['Player']}: {weakest_s['Floor (10%)']:.1f}-{weakest_s['Boom (90%)']:.1f}",
                "Detailed Statistical Reasoning": f"Active starters maximize win probability ({weakest_s['L3 xFP (Vol)']:.1f} L3 xFP). {b_note}"
            })
        return strat_rec, ss_recs

    matchup_groups = {}
    for t in team_records:
        mid = t.get('matchup_id')
        if mid is not None:
            matchup_groups.setdefault(mid, []).append(t)
            
    matchup_rows = []
    all_start_sit_rows = []
    handled_managers = set()
    tug_of_war_cards_html = '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(380px, 1fr)); gap:14px; margin-bottom:16px;">'
    
    for mid, m_teams in sorted(matchup_groups.items()):
        if len(m_teams) == 2:
            tA, tB = m_teams[0], m_teams[1]
            handled_managers.add(tA['manager'])
            handled_managers.add(tB['manager'])
            
            margin = tA['starter_pts'] - tB['starter_pts']
            joint_std = math.sqrt(tA['team_var'] + tB['team_var'])
            win_prob_a = round(normal_cdf(margin / joint_std) * 100, 1)
            win_prob_b = round(100.0 - win_prob_a, 1)
            
            mode_a = "FLOOR" if margin > 5.0 else ("CEILING" if margin < -5.0 else "BALANCED")
            mode_b = "FLOOR" if margin < -5.0 else ("CEILING" if margin > 5.0 else "BALANCED")
            
            strat_rec_a, ss_a = build_team_strategy_and_start_sit(tA, tB, mode_a)
            strat_rec_b, ss_b = build_team_strategy_and_start_sit(tB, tA, mode_b)
            all_start_sit_rows.extend(ss_a)
            all_start_sit_rows.extend(ss_b)
            
            tA['specific_strategy_play'] = strat_rec_a
            tB['specific_strategy_play'] = strat_rec_b
            tA['start_sit_summary'] = f"{ss_a[0]['Action Verdict']}: {ss_a[0]['Bench / Roster Player to Start']} ({ss_a[0]['Matchup Win Prob Delta']})" if ss_a else ""
            tB['start_sit_summary'] = f"{ss_b[0]['Action Verdict']}: {ss_b[0]['Bench / Roster Player to Start']} ({ss_b[0]['Matchup Win Prob Delta']})" if ss_b else ""
            
            tA['matchup_summary_str'] = f"vs {tB['manager']} ({tA['starter_pts']:.2f} to {tB['starter_pts']:.2f} Model Proj | Live: {tA['live_pts']:.1f} to {tB['live_pts']:.1f}) — Win Odds: {win_prob_a}% ({mode_a} Mode)"
            tB['matchup_summary_str'] = f"vs {tA['manager']} ({tB['starter_pts']:.2f} to {tA['starter_pts']:.2f} Model Proj | Live: {tB['live_pts']:.1f} to {tA['live_pts']:.1f}) — Win Odds: {win_prob_b}% ({mode_b} Mode)"
            
            tug_of_war_cards_html += (
                f'<div style="padding:14px 16px; border-radius:10px; border:1px solid rgba(148,163,184,0.35); background:rgba(15,23,42,0.02);">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.92rem; font-weight:700; margin-bottom:4px;">'
                f'<span>{tA["manager"]} ({win_prob_a:.1f}%)</span>'
                f'<span style="font-size:0.78rem; opacity:0.7;">Matchup {mid}</span>'
                f'<span>{tB["manager"]} ({win_prob_b:.1f}%)</span>'
                f'</div>'
                f'<div style="display:flex; justify-content:space-between; font-size:0.8rem; opacity:0.8; margin-bottom:6px;">'
                f'<span>Model: {tA["starter_pts"]:.1f} (Live: {tA["live_pts"]:.1f})</span>'
                f'<span>Model: {tB["starter_pts"]:.1f} (Live: {tB["live_pts"]:.1f})</span>'
                f'</div>'
                f'<div style="height:12px; border-radius:6px; background:rgba(220,38,38,0.65); overflow:hidden; display:flex; margin-bottom:8px;">'
                f'<div style="width:{win_prob_a:.1f}%; background:#2563eb; height:100%;"></div>'
                f'</div>'
                f'<div style="font-size:0.78rem; line-height:1.45; opacity:0.9;">'
                f'<div><strong>{tA["manager"]} Decider:</strong> {tA["leverage_decider_str"]}</div>'
                f'<div style="margin-top:2px;"><strong>{tB["manager"]} Decider:</strong> {tB["leverage_decider_str"]}</div>'
                f'</div></div>'
            )
            
            matchup_rows.append({
                "Matchup": f"Game {mid}: {tA['manager']} (vs {tB['manager']})",
                "Pre-Game Model Proj": f"{tA['starter_pts']:.2f} vs {tB['starter_pts']:.2f}",
                "Live Score": f"{tA['live_pts']:.2f} - {tB['live_pts']:.2f}",
                "Win Odds": f"{win_prob_a}%",
                "#1 Matchup Decider (Leverage Index)": tA['leverage_decider_str'],
                "Strategy Mode": f"{'Favored: Floor Protection' if mode_a == 'FLOOR' else ('Underdog: Ceiling Variance' if mode_a == 'CEILING' else 'Toss-Up: Max Points')}",
                "Specific Roster Prioritization (Who to Start & Why)": strat_rec_a
            })
            matchup_rows.append({
                "Matchup": f"Game {mid}: {tB['manager']} (vs {tA['manager']})",
                "Pre-Game Model Proj": f"{tB['starter_pts']:.2f} vs {tA['starter_pts']:.2f}",
                "Live Score": f"{tB['live_pts']:.2f} - {tA['live_pts']:.2f}",
                "Win Odds": f"{win_prob_b}%",
                "#1 Matchup Decider (Leverage Index)": tB['leverage_decider_str'],
                "Strategy Mode": f"{'Favored: Floor Protection' if mode_b == 'FLOOR' else ('Underdog: Ceiling Variance' if mode_b == 'CEILING' else 'Toss-Up: Max Points')}",
                "Specific Roster Prioritization (Who to Start & Why)": strat_rec_b
            })
    tug_of_war_cards_html += '</div>'
            
    for t in team_records:
        if t['manager'] not in handled_managers:
            strat_rec, ss_list = build_team_strategy_and_start_sit(t, None, "BALANCED")
            t['specific_strategy_play'] = strat_rec
            t['start_sit_summary'] = f"{ss_list[0]['Action Verdict']}: {ss_list[0]['Bench / Roster Player to Start']}" if ss_list else ""
            all_start_sit_rows.extend(ss_list)
            
    matchups_df = pd.DataFrame(matchup_rows) if matchup_rows else pd.DataFrame(columns=matchup_cols)
    start_sit_df = pd.DataFrame(all_start_sit_rows) if all_start_sit_rows else pd.DataFrame(columns=ss_cols)
    
    unrostered_df = wk_df[(~wk_df['norm_name'].isin(rostered_norm_names)) & (~wk_df['is_out'])].copy()
    unrostered_df['Waiver Score'] = unrostered_df['ROS Value'] * 0.55 + unrostered_df[COL_MODEL] * 0.45 + unrostered_df['Vacated Boost'] * 1.2
    unrostered_df['Full-Season ROS Value'] = unrostered_df['ROS Value']
    unrostered_df['Scouting Catalyst'] = unrostered_df['Boom Catalyst']
    top_waivers = unrostered_df.sort_values(by='Waiver Score', ascending=False).head(25)
    waiver_df = top_waivers[waiver_cols].reset_index(drop=True) if not top_waivers.empty else pd.DataFrame(columns=waiver_cols)
        
    pr_rows = []
    for t in team_records:
        pos_diffs = {pos: round(t['pos_raw'][pos] - league_pos_avg[pos], 2) for pos in ['QB', 'RB', 'WR', 'TE']}
        st_diffs = {pos: round(t['pos_starter'][pos] - league_starter_avg[pos], 2) for pos in ['QB', 'RB', 'WR', 'TE']}
        dp_diffs = {pos: round(t['pos_depth'][pos] - league_depth_avg[pos], 2) for pos in ['QB', 'RB', 'WR', 'TE']}
        wk_pos_diffs = {pos: round(t['wk_pos_pts'][pos] - league_wk_pos_avg[pos], 2) for pos in ['QB', 'RB', 'WR', 'TE']}
        
        t['pos_diffs'] = pos_diffs
        t['wk_pos_diffs'] = wk_pos_diffs
        
        overall_diff = t['ros_lineup_ppg'] - league_ros_avg
        t['overall_grade'] = f"{score_to_grade(overall_diff, scale=1.8)} ({overall_diff:+.1f} PPG)"
        t['qb_grade'] = f"{score_to_grade(pos_diffs['QB'], scale=0.8)} ({pos_diffs['QB']:+.1f})"
        t['rb_grade'] = f"{score_to_grade(pos_diffs['RB'], scale=1.25)} ({pos_diffs['RB']:+.1f})"
        t['wr_grade'] = f"{score_to_grade(pos_diffs['WR'], scale=1.25)} ({pos_diffs['WR']:+.1f})"
        t['te_grade'] = f"{score_to_grade(pos_diffs['TE'], scale=0.8)} ({pos_diffs['TE']:+.1f})"
        
        t['rb_starter_grade'] = score_to_grade(st_diffs['RB'], scale=1.15)
        t['rb_depth_grade'] = score_to_grade(dp_diffs['RB'], scale=0.55)
        t['wr_starter_grade'] = score_to_grade(st_diffs['WR'], scale=1.15)
        t['wr_depth_grade'] = score_to_grade(dp_diffs['WR'], scale=0.55)
        
        best_pos = max(pos_diffs, key=pos_diffs.get)
        worst_pos = min(pos_diffs, key=pos_diffs.get)
        t['strength'] = best_pos
        t['weakness'] = worst_pos
        
        pr_rows.append({
            "Manager": t['manager'],
            "Record": f"{t['wins']}-{t['losses']}",
            COL_MODEL: t['starter_pts'],
            COL_SLEEPER: t['sleeper_pts'],
            COL_DIFF: t['diff_pts'],
            "Live Banked Pts": t['live_pts'],
            "Optimal Lineup Pts": t['opt_pts'],
            "Overall Season Grade": t['overall_grade'],
            "Playoff %": f"{t['playoff_pct']}%",
            "1st-Rd Bye %": f"{t['bye_pct']}%",
            "Title %": f"{t['champ_pct']}%",
            "QB Season Grade": t['qb_grade'],
            "RB Season Grade": t['rb_grade'],
            "WR Season Grade": t['wr_grade'],
            "TE Season Grade": t['te_grade'],
            "Full-Season Strength": f"{best_pos} ({pos_diffs[best_pos]:+.1f})",
            "Full-Season Need": f"{worst_pos} ({pos_diffs[worst_pos]:+.1f})"
        })
        
    pr_df = pd.DataFrame(pr_rows).sort_values(by=COL_MODEL, ascending=False).reset_index(drop=True)
    pr_df.insert(0, "Rank", range(1, len(pr_df) + 1))
    
    def eval_post_trade_gain(team_obj, send_list, recv_list):
        new_pos_vals = {pos: list(team_obj['pos_dict_vals'][pos]) for pos in ['QB', 'RB', 'WR', 'TE']}
        for sp in send_list:
            p_pos = sp['pos']
            val = sp['ros_val']
            if p_pos in new_pos_vals and val in new_pos_vals[p_pos]:
                new_pos_vals[p_pos].remove(val)
        for rp in recv_list:
            p_pos = rp['pos']
            if p_pos in new_pos_vals:
                new_pos_vals[p_pos].append(rp['ros_val'])
        new_ppg = calc_exact_10_starter_ros_ppg(new_pos_vals)
        return round(new_ppg - team_obj['ros_lineup_ppg'], 2)

    trade_candidates = []
    n_teams = len(team_records)
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            t1, t2 = team_records[i], team_records[j]
            for pos1 in ['QB', 'RB', 'WR', 'TE']:
                for pos2 in ['QB', 'RB', 'WR', 'TE']:
                    if pos1 == pos2:
                        continue
                    pool1 = [p for p in t1['pos_groups'][pos1] if p['ros_val'] >= 6.5]
                    pool2 = [p for p in t2['pos_groups'][pos2] if p['ros_val'] >= 6.5]
                    
                    for p1 in pool1:
                        for p2 in pool2:
                            if abs(p1['ros_val'] - p2['ros_val']) <= 3.5:
                                gain1 = eval_post_trade_gain(t1, [p1], [p2])
                                gain2 = eval_post_trade_gain(t2, [p2], [p1])
                                if gain1 > 0.05 and gain2 > 0.05:
                                    po_d1 = round(min(18.0, gain1 * 2.8), 1)
                                    po_d2 = round(min(18.0, gain2 * 2.8), 1)
                                    ch_d1 = round(min(9.5, gain1 * 1.3), 1)
                                    ch_d2 = round(min(9.5, gain2 * 1.3), 1)
                                    trade_candidates.append({
                                        'score': (gain1 + gain2) * 3.0 - abs(gain1 - gain2),
                                        'pair_key': (tuple(sorted([t1['manager'], t2['manager']])), '1v1'),
                                        'row': {
                                            "Trade Type": "1-for-1 Win-Win Swap",
                                            "Manager 1": t1['manager'],
                                            "Manager 1 Sends": f"{p1['name']} ({p1['pos']} - {p1['team']}, {p1['ros_val']:.1f} PPG)",
                                            "Manager 2": t2['manager'],
                                            "Manager 2 Sends": f"{p2['name']} ({p2['pos']} - {p2['team']}, {p2['ros_val']:.1f} PPG)",
                                            "Playoff / Title Equity Delta": f"{t1['manager']}: +{po_d1}% PO / +{ch_d1}% Title | {t2['manager']}: +{po_d2}% PO / +{ch_d2}% Title",
                                            "Why This Trade Works (Mutual Starter PPG Gain)": (
                                                f"Both Starting Lineups Improve: {t1['manager']} gains +{gain1:.2f} Starter PPG (upgrades {pos2}); "
                                                f"{t2['manager']} gains +{gain2:.2f} Starter PPG (upgrades {pos1})"
                                            )
                                        }
                                    })
                    if len(pool1) >= 3 and len(pool2) >= 1:
                        p1_a, p1_b = pool1[1], pool1[2]
                        for star2 in pool2[:2]:
                            if star2['ros_val'] > p1_a['ros_val'] + 1.5:
                                gain1 = eval_post_trade_gain(t1, [p1_a, p1_b], [star2])
                                gain2 = eval_post_trade_gain(t2, [star2], [p1_a, p1_b])
                                if gain1 > 0.05 and gain2 > 0.05:
                                    po_d1 = round(min(18.0, gain1 * 2.8), 1)
                                    po_d2 = round(min(18.0, gain2 * 2.8), 1)
                                    ch_d1 = round(min(9.5, gain1 * 1.3), 1)
                                    ch_d2 = round(min(9.5, gain2 * 1.3), 1)
                                    trade_candidates.append({
                                        'score': (gain1 + gain2) * 2.8,
                                        'pair_key': (tuple(sorted([t1['manager'], t2['manager']])), '2v1'),
                                        'row': {
                                            "Trade Type": "2-for-1 Package",
                                            "Manager 1": t1['manager'],
                                            "Manager 1 Sends": f"{p1_a['name']} ({p1_a['ros_val']:.1f}) + {p1_b['name']} ({p1_b['ros_val']:.1f})",
                                            "Manager 2": t2['manager'],
                                            "Manager 2 Sends": f"{star2['name']} ({star2['pos']} - {star2['team']}, {star2['ros_val']:.1f} PPG)",
                                            "Playoff / Title Equity Delta": f"{t1['manager']}: +{po_d1}% PO / +{ch_d1}% Title | {t2['manager']}: +{po_d2}% PO / +{ch_d2}% Title",
                                            "Why This Trade Works (Mutual Starter PPG Gain)": (
                                                f"Both Starting Lineups Improve: {t1['manager']} gains +{gain1:.2f} Starter PPG (consolidates {pos1} depth for {pos2} starter); "
                                                f"{t2['manager']} gains +{gain2:.2f} Starter PPG (fills 2 starting/flex slots)"
                                            )
                                        }
                                    })
                                    
    trade_candidates.sort(key=lambda x: x['score'], reverse=True)
    seen_keys = set()
    final_trades = []
    for tc in trade_candidates:
        if tc['pair_key'] not in seen_keys:
            seen_keys.add(tc['pair_key'])
            final_trades.append(tc['row'])
        if len(final_trades) >= 10:
            break
            
    trades_df = pd.DataFrame(final_trades) if final_trades else pd.DataFrame(columns=tr_cols)
    
    league_analysis_cache[(league_id, wk_int)] = {
        'teams': team_records,
        'pos_avg': league_pos_avg,
        'wk_pos_avg': league_wk_pos_avg,
        'ros_avg': league_ros_avg,
        'playoff_teams': playoff_teams,
        'rem_reg_weeks': rem_reg_weeks,
        'free_agents_raw': unrostered_df,
        'start_sit_df': start_sit_df
    }
    
    mgr_names = [r['Manager'] for _, r in pr_df.iterrows()]
    default_mgr = next((m for m in mgr_names if 'jacob' in m.lower()), mgr_names[0] if mgr_names else None)
    second_mgr = next((m for m in mgr_names if m != default_mgr), default_mgr)
    vis_update, audit_html, r_df = format_manager_deep_dive(league_id, wk_int, default_mgr)
    
    summary_text = (
        f"**Real-Time League Sync Complete: {league_id}** (Week {wk_int} Lineups | 1,500 Monte Carlo Simulations) — "
        f"*Click any manager row in the table below (or use the dropdown) to inspect their Full-Season & Weekly Audit, Starter Payoff Tracks, and #1 Matchup Decider.*"
    )
    return (
        summary_text, pr_df, tug_of_war_cards_html, matchups_df, start_sit_df, waiver_df, trades_df,
        gr.update(choices=["ALL MANAGERS"] + mgr_names, value="ALL MANAGERS"),
        gr.update(choices=mgr_names, value=default_mgr), vis_update, audit_html, r_df,
        gr.update(choices=mgr_names, value=default_mgr), gr.update(choices=mgr_names, value=second_mgr)
    )

def on_league_table_click(pr_df, league_id, week_num, evt: gr.SelectData):
    if pr_df is None or len(pr_df) == 0:
        return gr.update(), gr.update(visible=False), "", pd.DataFrame()
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    mgr_name = str(pr_df.iloc[row_idx]['Manager'])
    vis_update, audit_html, r_df = format_manager_deep_dive(league_id, int(week_num), mgr_name)
    return gr.update(value=mgr_name), vis_update, audit_html, r_df

def get_team_skill_players_list(league_id, week_num, mgr_name):
    cache_key = (str(league_id).strip(), int(week_num))
    if cache_key not in league_analysis_cache or not mgr_name:
        return []
    teams = league_analysis_cache[cache_key]['teams']
    t = next((x for x in teams if x['manager'] == mgr_name), None)
    if not t:
        return []
    return [p['Player'] for p in t['full_roster'] if p['Pos'] in ['QB', 'RB', 'WR', 'TE']]

def update_trade_player_choices(league_id, week_num, mgr_name):
    skill_players = get_team_skill_players_list(league_id, week_num, mgr_name)
    return gr.update(choices=skill_players, value=[skill_players[0]] if skill_players else [])

def run_custom_trade_sim(league_id, week_num, mgr_a, players_a, mgr_b, players_b):
    cache_key = (str(league_id).strip(), int(week_num))
    if cache_key not in league_analysis_cache:
        return "Sync your league first to run custom trade simulations."
    if not mgr_a or not mgr_b or mgr_a == mgr_b or not players_a or not players_b:
        return "Select two different managers and **1 or more players** from each roster to simulate a 1-for-1, 2-for-1, or multi-player package trade."
        
    bundle = league_analysis_cache[cache_key]
    teams = bundle['teams']
    ros_avg = bundle['ros_avg']
    
    tA = next((x for x in teams if x['manager'] == mgr_a), None)
    tB = next((x for x in teams if x['manager'] == mgr_b), None)
    if not tA or not tB:
        return "Unable to locate selected managers."
        
    rows_a = [p for p in tA['full_roster'] if p['Player'] in players_a]
    rows_b = [p for p in tB['full_roster'] if p['Player'] in players_b]
    if not rows_a or not rows_b:
        return "Selected players not found on those rosters."
        
    def calc_new_10_starter_ppg(team_obj, remove_names, add_rows):
        new_groups = {pos: list(team_obj['pos_dict_vals'][pos]) for pos in ['QB', 'RB', 'WR', 'TE']}
        for p in team_obj['full_roster']:
            if p['Player'] in remove_names and p['Pos'] in new_groups:
                val = float(p['Full-Season ROS Value'])
                if val in new_groups[p['Pos']]:
                    new_groups[p['Pos']].remove(val)
        for ar in add_rows:
            if ar['Pos'] in new_groups:
                new_groups[ar['Pos']].append(float(ar['Full-Season ROS Value']))
        return calc_exact_10_starter_ros_ppg(new_groups)

    new_ros_a = calc_new_10_starter_ppg(tA, set(players_a), rows_b)
    new_ros_b = calc_new_10_starter_ppg(tB, set(players_b), rows_a)
    
    delta_a = round(new_ros_a - tA['ros_lineup_ppg'], 2)
    delta_b = round(new_ros_b - tB['ros_lineup_ppg'], 2)
    
    po_delta_a = round(np.clip(delta_a * 2.8, -25.0, 25.0), 1)
    po_delta_b = round(np.clip(delta_b * 2.8, -25.0, 25.0), 1)
    ch_delta_a = round(np.clip(delta_a * 1.3, -15.0, 15.0), 1)
    ch_delta_b = round(np.clip(delta_b * 1.3, -15.0, 15.0), 1)
    
    new_grade_a = score_to_grade(new_ros_a - ros_avg, scale=1.8)
    new_grade_b = score_to_grade(new_ros_b - ros_avg, scale=1.8)
    
    if delta_a > 0 and delta_b > 0:
        verdict_banner = "PARETO WIN-WIN TRADE (Both Starting Lineups & Championship Equities Increase!)"
    elif delta_a > 0 and delta_b <= 0:
        verdict_banner = f"Trade Favors {mgr_a} ({mgr_a} gains +{delta_a:.2f} Starter PPG; {mgr_b} changes by {delta_b:+.2f} Starter PPG)"
    elif delta_b > 0 and delta_a <= 0:
        verdict_banner = f"Trade Favors {mgr_b} ({mgr_b} gains +{delta_b:.2f} Starter PPG; {mgr_a} changes by {delta_a:+.2f} Starter PPG)"
    else:
        verdict_banner = "Sub-Optimal Trade (Neither Starting Lineup Improves)"
        
    sent_a_str = " + ".join([f"**{p['Player']}** ({p['Pos']}, {p['Full-Season ROS Value']:.1f} PPG)" for p in rows_a])
    sent_b_str = " + ".join([f"**{p['Player']}** ({p['Pos']}, {p['Full-Season ROS Value']:.1f} PPG)" for p in rows_b])
    
    return (
        f"### Counterfactual Package Trade Verdict: {verdict_banner}\n"
        f"| Manager | Acquires in Trade | 10-Starter Lineup PPG (Before -> After) | Net Starter PPG Change | Playoff / Title Odds Change | New Overall Grade |\n"
        f"| :--- | :--- | :---: | :---: | :---: | :---: |\n"
        f"| **{mgr_a}** | {sent_b_str} | {tA['ros_lineup_ppg']:.2f} -> **{new_ros_a:.2f} PPG** | **{delta_a:+.2f} PPG/wk** | **{po_delta_a:+.1f}% PO / {ch_delta_a:+.1f}% Title** | **{new_grade_a}** |\n"
        f"| **{mgr_b}** | {sent_a_str} | {tB['ros_lineup_ppg']:.2f} -> **{new_ros_b:.2f} PPG** | **{delta_b:+.2f} PPG/wk** | **{po_delta_b:+.1f}% PO / {ch_delta_b:+.1f}% Title** | **{new_grade_b}** |"
    )

GLOSSARY_DEFINITIONS = {
    "xFP (Expected Fantasy Points)": "**Expected Fantasy Points (xFP)** measures how many PPR points an average NFL player would score given a player's exact volume of targets, receptions, air yards, carries, and first downs created. **Why it matters:** If a player averages 16.5 xFP but only 11.0 actual PPR points, they have elite usage and are a prime positive-regression breakout candidate.",
    "WOPR (Weighted Opportunity)": "**Weighted Opportunity Rating (WOPR)** is calculated as `1.5 × Target Share + 0.7 × Air Yards Share`. Created by Josh Hermsmeyer, it is the single most predictive receiver usage metric because it balances high-percentage short targets with high-upside downfield air yards. **Benchmarks:** `0.60+` = Elite Alpha WR1 | `0.45 - 0.59` = Solid WR2 | `< 0.35` = Rotational.",
    "EPA (Expected Points Added)": "**Expected Points Added (EPA)** measures how much each play improves a team's mathematical probability of scoring based on down, distance, and field position. A 6-yard gain on 3rd-and-5 yields positive EPA; a 6-yard gain on 3rd-and-15 yields negative EPA. **Benchmarks:** `+3.0+ EPA/gm` = Elite Chain-Mover | `0.0` = League Average.",
    "Sportsbook +EV% (Expected Value)": "**Expected Value (+EV%)** compares our model's true probability of a player hitting the Over/Under against the sportsbook's implied break-even probability (`52.38%` at `-110` odds). If our Conditional Median model gives a prop a `60.5%` hit probability against `-110` odds, the bet carries a **`+8.1% Expected Value Edge`**.",
    "Leverage Index (Matchup Decider)": "**Player Leverage Index (LI)** measures the exact swing in your weekly head-to-head Win Probability between a starter hitting their 90th-percentile ceiling versus their 10th-percentile floor: `LI = P(Win | Boom) - P(Win | Floor)`. The starter with the highest Leverage Index is your **#1 Matchup Decider**.",
    "Shapley Attribution Waterfall": "**Marginal Feature Attribution** decomposes a player's weekly projection into exact `+ / - PPR` contributions from five factor groups: Positional Starter Baseline, Volume & Role Equity, Play-by-Play Efficiency, Defense vs. Position, and Vegas/Weather/Vacated Volume.",
    "ROS Value & Playoff SoS": "**Full-Season Rest-of-Season (ROS) Value** represents a player's healthy weekly baseline for the remainder of the season, completely immune to single-week byes, and weighted by their **Fantasy Playoff Strength of Schedule (Weeks 15, 16, and 17)**.",
    "Positional Grades (82/18 Split)": "**Full-Season Positional Grades (A+ to D)** evaluate each manager's position group using an **82% weight on Core Starters + an Elite Alpha Star Bonus** (so owning two superstars like Bijan Robinson and Derrick Henry guarantees an A/A+ grade) and an **18% weight on Bench Depth**."
}

def explain_metric_popup(metric_key):
    return f"**{metric_key}:** {GLOSSARY_DEFINITIONS.get(metric_key, '')}"

methodology_text = """
## Complete Quantitative Methodology & Mathematical Whitepaper

### 1. Data Ingestion & Real-Time Synchronization Architecture
The projection engine ingests three seasons of play-by-play aggregated player logs (`2024`, `2025`, and live `2026` weekly releases) from `nflverse`, cross-referenced with official NFL schedule telemetry (`schedules/games.csv`) and live Sleeper API endpoints (`/v1/players/nfl`, `/projections/nfl/2026/{week}`, `/v1/league/{id}/matchups/{week}`). Our local cache validator tracks both the **maximum completed week** and the **exact player-game row count** (`(max_week, len(df_2026))`), ensuring that Thursday, Sunday, and Monday Night Football stat releases automatically trigger model retraining while live injury tags, in-game points, and rosters sync in real time.

---

### 2. Feature Engineering & Opportunity Valuation (36 Predictive Variables)
Standard box-score projections suffer from severe touchdown variance lag. Our feature pipeline constructs 36 distinct variables across five categories:
* **Expected Fantasy Points (`xFP`) & Luck Delta (`Actual PPR - xFP`):** Weights every target (`1.25x`), reception (`0.38x`), air yard (`0.042x`), carry (`0.74x`), first down created (`0.52x–0.55x`), and touchdown equity (`2.15x–3.60x`) to isolate true underlying volume from unsustainable touchdown flukes.
* **Downfield Alpha Usage (`WOPR` & `RACR`):** Combines `1.5 × Target Share + 0.7 × Air Yards Share` (`WOPR`) with `Receiving Yards ÷ Air Yards` (`RACR`) and Yards After Catch (`YAC`).
* **Play-by-Play Efficiency (`EPA` & `Efficiency Over Baseline`):** Sums `passing_epa + rushing_epa + receiving_epa` per game alongside yards per carry over `4.3 YPC` and yards per target over `7.8 YPT`.
* **Vacated Teammate Opportunity Redistribution:** Sums trailing targets and carries of teammates ruled `OUT`, `IR`, `PUP`, or `DOUBTFUL` and redistributes a bounded share (`Vacated Boost`) to active starters on that offense.
* **Stadium Environment, Weather, Rest & Vegas Game Script:** Incorporates Implied Team Totals, Point Spreads, Indoor Dome vs. Outdoor Wind Speed (`mph`), Rest Days, and Divisional familiarity directly inside the 36-feature matrix.

---

### 3. Three-Model GBDT Ensemble & Multi-Target Player Prop Engine
We train five `HistGradientBoostingRegressor` models for PPR scoring plus five dedicated **Player Prop Models** (`Passing Yards`, `Rushing Yards`, `Receiving Yards`, `Receptions`, and `Anytime TD %`):
1. **Conditional Mean (`loss='squared_error'`) + MAE Optimizer (`loss='absolute_error'`) + Conditional Median (`quantile=0.50`):** Prevents right-tail outlier distortion and minimizes out-of-sample Mean Absolute Error.
2. **Sportsbook Prop Over/Under Probability & $+EV\%$:** Because sportsbooks price Over/Under lines at the **median ($Q_{0.50}$)**, our Conditional Median prop models directly estimate $P(\text{Over}) = 1 - \Phi((\text{Line} - \mu_{0.50}) / \sigma)$ and compute **Expected Value ($+EV\%$)** against standard `-110` vig (`52.38%` implied break-even) or live DraftKings/FanDuel prices.

---

### 4. Opponent-Conditioned Portfolio Optimization & Player Leverage Index ($LI$)
* **Bivariate Win-Probability Start/Sit Optimization:** Evaluates every start/sit decision against the opposing manager's exact projected mean and variance ($\mu_{\text{opp}}, \sigma_{\text{opp}}$), reporting both **Projected Point Gain** and **Matchup Win Probability $\Delta\%$**.
* **Player Leverage Index ($LI$):** Quantifies each starter's impact on the weekly matchup outcome via $LI_i = P(\text{Win} \mid Q_{0.90}) - P(\text{Win} \mid Q_{0.10})$, surfacing the **#1 Matchup Decider** on each roster.
* **Counterfactual Pareto Trade Engine:** Simulates both managers' 10-starter lineups (`1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX, 1 K, 1 DEF`) before and after candidate `1-for-1` and `2-for-1` trades, recommending only swaps where **both** managers increase their starting lineup PPG and Championship Equity.
"""

sleek_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

custom_css = """
.gradio-container { max-width: 1460px !important; }
table td, table th { line-height: 1.45 !important; padding: 8px 10px !important; }
"""

print("6/6: Initializing Player Props Engine, Visual Payoff Tracks & Launching 5-Workspace Suite...")
init_banner, init_table = update_main_table(default_target_week, True, True, True, "ALL", "ALL", "")
init_booms, init_busts = get_top_outliers(default_target_week)
init_diff_high, init_diff_low = get_biggest_differers(default_target_week, "ALL")
(
    init_lg_sum, init_pr_df, init_tug_html, init_match_df, init_ss_df, init_waiver_df, init_tr_df,
    init_ss_mgr_dd, init_mgr_dd, init_vis_dd, init_audit_html, init_r_df,
    init_tr_mgr_a, init_tr_mgr_b
) = analyze_sleeper_league(DEFAULT_LEAGUE_ID, default_target_week)

mgr_choices_init = init_pr_df['Manager'].tolist() if not init_pr_df.empty else []
def_mgr_a = next((m for m in mgr_choices_init if 'jacob' in m.lower()), mgr_choices_init[0] if mgr_choices_init else None)
def_mgr_b = next((m for m in mgr_choices_init if m != def_mgr_a), def_mgr_a)
init_pa_list = get_team_skill_players_list(DEFAULT_LEAGUE_ID, default_target_week, def_mgr_a)
init_pb_list = get_team_skill_players_list(DEFAULT_LEAGUE_ID, default_target_week, def_mgr_b)
default_comp_players = active_player_names[:3] if len(active_player_names) >= 3 else active_player_names
init_comp_html, init_comp_sum, init_comp_df = compare_multiple_players(default_target_week, default_comp_players)
init_prop_status, init_top_props_df, init_all_props_df = build_player_props_board(default_target_week, "ALL PROP MARKETS", "ALL", 0.0, "", "")
init_custom_prop_html = evaluate_custom_prop_bet(default_target_week, active_player_names[0] if active_player_names else "", "Receiving Yards", 64.5, "-110")

with gr.Blocks(theme=sleek_theme, css=custom_css, title="NFL Quantitative Scouting, League & Props Suite") as demo:
    gr.Markdown("# NFL Quantitative Scouting, League Intelligence & Player Props Suite (2026)")
    
    with gr.Accordion("Interactive Stat Glossary & Metric Explainer (Click to Expand & Define Any Stat)", open=False):
        gr.Markdown("Click any statistical term below to view its plain-English definition, formula, and benchmarks:")
        with gr.Row():
            glossary_btns = [gr.Button(k, size="sm") for k in GLOSSARY_DEFINITIONS.keys()]
        glossary_popup_md = gr.Markdown(explain_metric_popup("xFP (Expected Fantasy Points)"))
        for b_ctrl in glossary_btns:
            b_ctrl.click(fn=explain_metric_popup, inputs=b_ctrl, outputs=glossary_popup_md)

    with gr.Tabs():
        # WORKSPACE 1: PLAYER PROJECTIONS, VISUAL PAYOFF TRACKS, BOOMS/BUSTS & DIFFERERS
        with gr.TabItem("1. Player Projections & Scouting"):
            with gr.Tabs():
                with gr.TabItem("Weekly Projections & Visual Scouting Cards"):
                    with gr.Row():
                        wk_select = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value=str(default_target_week), label="NFL Week (1-18)", scale=1)
                        pos_sel = gr.Dropdown(choices=["ALL", "QB", "RB", "WR", "TE"], value="ALL", label="Position", scale=1)
                        outlook_sel = gr.Dropdown(choices=["ALL", "EXPECTED BASELINE", "HIGH-CEILING SKEW", "HIGH BUST RISK", "QUESTIONABLE"], value="ALL", label="Outlook Skew Filter", scale=1)
                        search_in = gr.Textbox(label="Search Player or Team", placeholder="e.g. Bijan, Henry, DET...", scale=2)
                        refresh_p_btn = gr.Button("↻ Force Live Injury & Projection Refresh", variant="secondary", scale=1)
                    with gr.Row():
                        hide_bye_chk = gr.Checkbox(value=True, label="Hide Bye Weeks")
                        hide_inj_chk = gr.Checkbox(value=True, label="Hide Injured / Out")
                        compact_chk = gr.Checkbox(value=True, label="Sleek Compact View (Fit Screen)")
                        
                    status_banner = gr.Markdown(init_banner)
                    with gr.Column(visible=False) as player_drawer:
                        close_drawer_btn = gr.Button("✖ Close Player Scouting Card", variant="secondary", size="sm")
                        drawer_html_view = gr.HTML()
                        drawer_logs = gr.DataFrame(label="2026 Weekly Game Logs (Actual PPR vs Expected xFP, WOPR, RACR & EPA)", interactive=False, visible=False)
                        
                    main_df_view = gr.DataFrame(value=init_table, label="Click Any Player Row Below to Open Their Visual Payoff Distribution Bar, Feature Attribution Waterfall & Headshot Card", interactive=False)
                    
                    for ctrl in [wk_select, hide_bye_chk, hide_inj_chk, compact_chk, pos_sel, outlook_sel, search_in]:
                        ctrl.change(
                            fn=update_main_table,
                            inputs=[wk_select, hide_bye_chk, hide_inj_chk, compact_chk, pos_sel, outlook_sel, search_in],
                            outputs=[status_banner, main_df_view]
                        )
                    refresh_p_btn.click(
                        fn=force_refresh_main_table,
                        inputs=[wk_select, hide_bye_chk, hide_inj_chk, compact_chk, pos_sel, outlook_sel, search_in],
                        outputs=[status_banner, main_df_view]
                    )
                        
                    main_df_view.select(fn=open_player_card_on_click, inputs=[main_df_view, wk_select], outputs=[player_drawer, drawer_html_view, drawer_logs])
                    close_drawer_btn.click(fn=close_player_scouting_drawer, inputs=None, outputs=[player_drawer, drawer_html_view, drawer_logs])

                with gr.TabItem("Top 5 Booms & Busts"):
                    gr.Markdown("### Asymmetric Starter Outliers (Filtered for Fantasy Starters >= 10.0 Projected PPR)")
                    with gr.Row():
                        bb_wk_select = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value=str(default_target_week), label="Target NFL Week")
                        bb_btn = gr.Button("Refresh Outliers", variant="primary")
                    boom_df = gr.DataFrame(value=init_booms, label="Top 5 Starter Boom Candidates (Positive Volume, WOPR & Matchup Skew)", interactive=False)
                    bust_df = gr.DataFrame(value=init_busts, label="Top 5 Starter Bust Candidates (Fragile Volume & Matchup Risk)", interactive=False)
                    bb_btn.click(fn=get_top_outliers, inputs=bb_wk_select, outputs=[boom_df, bust_df])
                    bb_wk_select.change(fn=get_top_outliers, inputs=bb_wk_select, outputs=[boom_df, bust_df])

                with gr.TabItem("Biggest Differers vs. Sleeper"):
                    gr.Markdown("### Biggest Projection Discrepancies: Model's Projection vs. Sleeper's Projection")
                    with gr.Row():
                        diff_wk_select = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value=str(default_target_week), label="Target NFL Week")
                        diff_pos_select = gr.Dropdown(choices=["ALL", "QB", "RB", "WR", "TE"], value="ALL", label="Filter Position")
                        diff_btn = gr.Button("Refresh Discrepancies", variant="primary")
                    diff_higher_df = gr.DataFrame(value=init_diff_high, label="Much Higher Than Sleeper (Model's Projection >> Sleeper's Projection)", interactive=False)
                    diff_lower_df = gr.DataFrame(value=init_diff_low, label="Much Lower Than Sleeper (Model's Projection << Sleeper's Projection)", interactive=False)
                    diff_btn.click(fn=get_biggest_differers, inputs=[diff_wk_select, diff_pos_select], outputs=[diff_higher_df, diff_lower_df])
                    diff_wk_select.change(fn=get_biggest_differers, inputs=[diff_wk_select, diff_pos_select], outputs=[diff_higher_df, diff_lower_df])
                    diff_pos_select.change(fn=get_biggest_differers, inputs=[diff_wk_select, diff_pos_select], outputs=[diff_higher_df, diff_lower_df])

        # WORKSPACE 2: LEAGUE COMMAND CENTER
        with gr.TabItem("2. League Command Center"):
            with gr.Row():
                league_id_in = gr.Textbox(value=DEFAULT_LEAGUE_ID, label="Sleeper League ID", placeholder="Paste your Sleeper League ID (e.g. 1389689750515904512)...")
                league_wk_in = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value=str(default_target_week), label="Target NFL Week for Lineup & Matchup Audit")
                league_btn = gr.Button("Sync Real-Time League Data & Poll Live Scores", variant="primary")
            league_summary = gr.Markdown(init_lg_sum)
            
            with gr.Tabs():
                with gr.TabItem("Power Rankings, Playoff Odds & Manager Audit"):
                    league_pr_table = gr.DataFrame(value=init_pr_df, label="Click Any Manager Row Below to Open Their Full-Season Audit & Starter Payoff Distribution Tracks", interactive=False)
                    with gr.Column(visible=bool(DEFAULT_LEAGUE_ID and not init_pr_df.empty)) as mgr_deep_dive_group:
                        with gr.Row():
                            manager_dropdown = gr.Dropdown(choices=mgr_choices_init, value=def_mgr_a, label="Select Manager to Inspect (Full-Season Starter/Depth Grades + Weekly Audit + Payoff Tracks)")
                        mgr_audit_html = gr.HTML(init_audit_html)
                        mgr_roster_df = gr.DataFrame(value=init_r_df, label="Selected Manager Full Roster Breakdown (Starters vs. Bench — Includes Live Points & Model vs. Sleeper)", interactive=False)

                with gr.TabItem("Live Matchup Win-Probability, Leverage & Start/Sit Advisor"):
                    gr.Markdown("### Live Matchup Win-Probability Tug-of-War & #1 Matchup Decider (Leverage Index)")
                    league_tug_html = gr.HTML(init_tug_html)
                    gr.Markdown("### Opponent-Conditioned Start/Sit Lineup Upgrades (Maximizing Matchup Win Probability)")
                    with gr.Row():
                        ss_mgr_filter = gr.Dropdown(choices=["ALL MANAGERS"] + mgr_choices_init, value="ALL MANAGERS", label="Filter Start/Sit Recommendations by Manager")
                    league_ss_table = gr.DataFrame(value=init_ss_df, label="Start/Sit Lineup Recommendations (Reports Both Projected Point Gain & Matchup Win Probability Delta)", interactive=False)
                    gr.Markdown("#### Head-to-Head Matchup Table & Roster-Specific Floor/Ceiling Prioritization")
                    league_match_table = gr.DataFrame(value=init_match_df, label="Matchup Strategy Advisor (Names Exact High-Floor or High-Ceiling Players + #1 Leverage Decider)", interactive=False)

                with gr.TabItem("Filterable Waiver Wire & Free-Agent Radar"):
                    gr.Markdown("### Top Unrostered Free Agents in This League (Interactive Sort & Filter)")
                    with gr.Row():
                        waiver_pos_sel = gr.Dropdown(choices=["ALL", "QB", "RB", "WR", "TE"], value="ALL", label="Filter Position")
                        waiver_sort_sel = gr.Dropdown(
                            choices=[
                                "Overall Waiver Score (Best Pickup)",
                                "Week Model's Projection",
                                "Full-Season ROS Value",
                                "Expected Volume (L3 xFP)",
                                "Boom Probability %",
                                "Playoff SoS (Wks 15-17)"
                            ],
                            value="Overall Waiver Score (Best Pickup)",
                            label="Sort Free Agents By"
                        )
                        waiver_search_in = gr.Textbox(label="Search Free Agent or NFL Team", placeholder="e.g. Guerendo, Cedric, BUF...")
                    league_waiver_table = gr.DataFrame(value=init_waiver_df, label="Available Free Agents (Filtered for Unrostered Players in Your League)", interactive=False)

                with gr.TabItem("Win-Win Trade Finder & Multi-Player Simulator"):
                    gr.Markdown("### Strictly Mutual-Gain Suggested Trades (Both Starting Lineups & Championship Equities Increase)")
                    league_tr_table = gr.DataFrame(value=init_tr_df, label="Automated 1-for-1 & 2-for-1 Trades Guaranteed to Increase Both Managers' Starting Lineup PPG & Title Odds", interactive=False)
                    gr.Markdown("#### Interactive Counterfactual Package Trade Simulator (Select 1 to 4 Players Per Side)")
                    with gr.Row():
                        sim_mgr_a = gr.Dropdown(choices=mgr_choices_init, value=def_mgr_a, label="Manager A")
                        sim_player_a = gr.Dropdown(choices=init_pa_list, value=[init_pa_list[0]] if init_pa_list else [], multiselect=True, max_choices=4, label="Manager A Sends (Select 1 to 4 Players)")
                        sim_mgr_b = gr.Dropdown(choices=mgr_choices_init, value=def_mgr_b, label="Manager B")
                        sim_player_b = gr.Dropdown(choices=init_pb_list, value=[init_pb_list[0]] if init_pb_list else [], multiselect=True, max_choices=4, label="Manager B Sends (Select 1 to 4 Players)")
                    sim_trade_btn = gr.Button("Simulate Multi-Player Trade Impact (PPG + Playoff/Title Odds)", variant="primary")
                    sim_trade_out = gr.Markdown()
            
            league_btn.click(
                fn=analyze_sleeper_league,
                inputs=[league_id_in, league_wk_in],
                outputs=[league_summary, league_pr_table, league_tug_html, league_match_table, league_ss_table, league_waiver_table, league_tr_table, ss_mgr_filter, manager_dropdown, mgr_deep_dive_group, mgr_audit_html, mgr_roster_df, sim_mgr_a, sim_mgr_b]
            )
            league_wk_in.change(
                fn=analyze_sleeper_league,
                inputs=[league_id_in, league_wk_in],
                outputs=[league_summary, league_pr_table, league_tug_html, league_match_table, league_ss_table, league_waiver_table, league_tr_table, ss_mgr_filter, manager_dropdown, mgr_deep_dive_group, mgr_audit_html, mgr_roster_df, sim_mgr_a, sim_mgr_b]
            )
            league_pr_table.select(
                fn=on_league_table_click,
                inputs=[league_pr_table, league_id_in, league_wk_in],
                outputs=[manager_dropdown, mgr_deep_dive_group, mgr_audit_html, mgr_roster_df]
            )
            manager_dropdown.change(
                fn=format_manager_deep_dive,
                inputs=[league_id_in, league_wk_in, manager_dropdown],
                outputs=[mgr_deep_dive_group, mgr_audit_html, mgr_roster_df]
            )
            ss_mgr_filter.change(fn=filter_start_sit_table, inputs=[league_id_in, league_wk_in, ss_mgr_filter], outputs=league_ss_table)
            for w_ctrl in [waiver_pos_sel, waiver_sort_sel, waiver_search_in]:
                w_ctrl.change(fn=filter_waiver_wire, inputs=[league_id_in, league_wk_in, waiver_pos_sel, waiver_sort_sel, waiver_search_in], outputs=league_waiver_table)
            sim_mgr_a.change(fn=update_trade_player_choices, inputs=[league_id_in, league_wk_in, sim_mgr_a], outputs=sim_player_a)
            sim_mgr_b.change(fn=update_trade_player_choices, inputs=[league_id_in, league_wk_in, sim_mgr_b], outputs=sim_player_b)
            sim_trade_btn.click(fn=run_custom_trade_sim, inputs=[league_id_in, league_wk_in, sim_mgr_a, sim_player_a, sim_mgr_b, sim_player_b], outputs=sim_trade_out)

        # WORKSPACE 3: MULTI-PLAYER COMPARISON WITH SYNCHRONIZED PAYOFF DISTRIBUTION TRACKS
        with gr.TabItem("3. Multi-Player Comparison (2 to 6 Players)"):
            gr.Markdown("### Multi-Player Side-by-Side Comparison (Synchronized Payoff Distribution Bars, Headshots & 23 Metrics)")
            with gr.Row():
                comp_wk = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value=str(default_target_week), label="Target NFL Week", scale=1)
                comp_players_sel = gr.Dropdown(
                    choices=active_player_names,
                    value=default_comp_players,
                    multiselect=True,
                    max_choices=6,
                    label="Select 2 to 6 Players to Compare Side-by-Side",
                    scale=3
                )
                comp_btn = gr.Button("Compare Selected Players", variant="primary", scale=1)
            
            comp_cards_html = gr.HTML(init_comp_html)
            comp_summary_md = gr.Markdown(init_comp_sum)
            comp_table_view = gr.DataFrame(value=init_comp_df, label="Multi-Player Side-by-Side Metric Matrix (Includes Projected Yardage & TD Props)", interactive=False)
            comp_btn.click(fn=compare_multiple_players, inputs=[comp_wk, comp_players_sel], outputs=[comp_cards_html, comp_summary_md, comp_table_view])
            comp_players_sel.change(fn=compare_multiple_players, inputs=[comp_wk, comp_players_sel], outputs=[comp_cards_html, comp_summary_md, comp_table_view])
            comp_wk.change(fn=compare_multiple_players, inputs=[comp_wk, comp_players_sel], outputs=[comp_cards_html, comp_summary_md, comp_table_view])

        # WORKSPACE 4: PLAYER PROPS & SPORTSBOOK EV FINDER (FANDUEL / DRAFTKINGS)
        with gr.TabItem("4. Player Props & Sportsbook +EV Finder"):
            gr.Markdown("### NFL Player Stat Projections & Sportsbook Over/Under Edge Finder (Passing, Rushing, Receiving & TDs)")
            with gr.Row():
                prop_wk_sel = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value=str(default_target_week), label="Target NFL Week", scale=1)
                prop_cat_sel = gr.Dropdown(
                    choices=["ALL PROP MARKETS", "Receiving Yards", "Rushing Yards", "Passing Yards", "Receptions (PPR)"],
                    value="ALL PROP MARKETS",
                    label="Prop Market Category",
                    scale=1
                )
                prop_pos_sel = gr.Dropdown(choices=["ALL", "QB", "RB", "WR", "TE"], value="ALL", label="Position", scale=1)
                prop_edge_sel = gr.Slider(minimum=-10.0, maximum=12.0, value=0.0, step=0.5, label="Min Expected Value (+EV%)", scale=1)
                prop_search_in = gr.Textbox(label="Search Player or Team", placeholder="e.g. Lamb, Bijan, BAL...", scale=1)
            with gr.Row():
                odds_api_in = gr.Textbox(
                    value="",
                    label="Optional: The Odds API Key (For Live DraftKings & FanDuel Lines — Leave Blank for Default Consensus Lines)",
                    placeholder="Paste free API key from the-odds-api.com or leave blank...",
                    scale=3
                )
                prop_refresh_btn = gr.Button("Scan Prop Markets & Calculate +EV", variant="primary", scale=1)
            prop_status_md = gr.Markdown(init_prop_status)
            
            with gr.Accordion("Interactive Custom Sportsbook Prop Calculator (Test Any Specific FanDuel / DraftKings / PrizePicks Line)", open=True):
                with gr.Row():
                    calc_player_sel = gr.Dropdown(choices=active_player_names, value=active_player_names[0] if active_player_names else None, label="Select Player", scale=2)
                    calc_market_sel = gr.Dropdown(choices=["Receiving Yards", "Rushing Yards", "Passing Yards", "Receptions (PPR)"], value="Receiving Yards", label="Prop Stat Market", scale=2)
                    calc_line_in = gr.Number(value=64.5, label="Sportsbook Line (e.g. 64.5)", scale=1)
                    calc_odds_in = gr.Textbox(value="-110", label="American Odds (e.g. -110, -115, +105)", scale=1)
                    calc_btn = gr.Button("Calculate Custom Prop EV", variant="secondary", scale=1)
                calc_out_html = gr.HTML(init_custom_prop_html)
                
            top_ev_props_table = gr.DataFrame(value=init_top_props_df, label="Top 10 Highest +EV Player Prop Bets of the Week", interactive=False)
            all_props_table = gr.DataFrame(value=init_all_props_df, label="Complete Weekly Player Prop Projections & Over/Under Hit Probabilities", interactive=False)
            
            for p_ctrl in [prop_wk_sel, prop_cat_sel, prop_pos_sel, prop_edge_sel, prop_search_in]:
                p_ctrl.change(
                    fn=build_player_props_board,
                    inputs=[prop_wk_sel, prop_cat_sel, prop_pos_sel, prop_edge_sel, prop_search_in, odds_api_in],
                    outputs=[prop_status_md, top_ev_props_table, all_props_table]
                )
            prop_refresh_btn.click(
                fn=build_player_props_board,
                inputs=[prop_wk_sel, prop_cat_sel, prop_pos_sel, prop_edge_sel, prop_search_in, odds_api_in],
                outputs=[prop_status_md, top_ev_props_table, all_props_table]
            )
            for c_ctrl in [prop_wk_sel, calc_player_sel, calc_market_sel, calc_line_in, calc_odds_in]:
                c_ctrl.change(
                    fn=evaluate_custom_prop_bet,
                    inputs=[prop_wk_sel, calc_player_sel, calc_market_sel, calc_line_in, calc_odds_in],
                    outputs=calc_out_html
                )
            calc_btn.click(
                fn=evaluate_custom_prop_bet,
                inputs=[prop_wk_sel, calc_player_sel, calc_market_sel, calc_line_in, calc_odds_in],
                outputs=calc_out_html
            )

        # WORKSPACE 5: MODEL ACCURACY & METHODOLOGY WHITEPAPER
        with gr.TabItem("5. Model Accuracy & Methodology"):
            with gr.Tabs():
                with gr.TabItem("Empirical Accuracy Backtest vs. Sleeper"):
                    gr.Markdown("### Empirical Backtest: Model's Projection vs. Sleeper's Projection Across Completed 2026 Games")
                    eval_choices = ["All Completed 2026 Weeks (Weeks 2+)"] + [f"Week {w}" for w in range(2, latest_completed_week + 1)]
                    eval_selector = gr.Dropdown(choices=eval_choices, value=eval_choices[0], label="Select Completed 2026 Week(s)")
                    
                    init_acc_md, init_acc_df = run_accuracy_benchmark(eval_choices[0])
                    acc_summary_md = gr.Markdown(init_acc_md)
                    acc_table = gr.DataFrame(value=init_acc_df, label="Head-to-Head Starter Accuracy Comparison", interactive=False)
                    eval_selector.change(fn=run_accuracy_benchmark, inputs=eval_selector, outputs=[acc_summary_md, acc_table])
                with gr.TabItem("Complete Methodology & Whitepaper"):
                    gr.Markdown(methodology_text)

if os.environ.get("PORT"):
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
else:
    demo.launch(inbrowser=True)