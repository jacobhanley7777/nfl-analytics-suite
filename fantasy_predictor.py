import os
import gc
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

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache.pkl")
PLAYERS_CACHE_PATH = os.path.expanduser("~/.sleeper_players_2026_cache.pkl")
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

team_fix = {'LAR': 'LA', 'WSH': 'WAS', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA'}
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

STATE = {
    'ready': False, 'schedule_map': {}, 'sleeper_pid_to_info': {},
    'norm_to_bio': {}, 'gsis_to_bio': {}, 'df': None, 'df_2026': None,
    'latest_dvp': {}, 'model_mean': None, 'model_median': None, 'model_mae': None,
    'model_floor': None, 'model_boom': None, 'prop_models': {}, 'pos_baselines': {},
    'name_col': 'player_display_name', 'team_col': 'team', 'opp_col': 'opponent_team',
    'latest_completed_week': 10, 'default_target_week': 11, 'active_player_names': [],
    'playoff_sos_map': {}, 'optimal_ml_weight': 0.60
}

sleeper_cache = {}
projection_week_cache = {}
league_analysis_cache = {}
live_sportsbook_props_cache = {}

def norm_name(n):
    if not isinstance(n, str):
        return ""
    n = re.sub(r'[^a-zA-Z\s]', '', n.lower())
    for sfx in [' jr', ' sr', ' iii', ' ii', ' iv', ' v']:
        if n.endswith(sfx):
            n = n[:-len(sfx)]
    n = re.sub(r'\s+', ' ', n).strip()
    return NICKNAME_MAP.get(n, n)

def ensure_engine_initialized():
    if STATE['ready']:
        return
    print("Initializing NFL Schedules, Player Bios & Pre-Trained Ensemble...")
    if not os.path.exists(CACHE_PATH):
        try:
            rel_url = "https://github.com/jacobhanley7777/nfl-analytics-suite/releases/download/v1.0/model_cache.pkl"
            r_dl = requests.get(rel_url, timeout=30, allow_redirects=True)
            if r_dl.status_code == 200:
                with open(CACHE_PATH, "wb") as f_out:
                    f_out.write(r_dl.content)
        except Exception as e:
            print(f"Release cache download notice: {e}")

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

        h_dict = {'opp': a_team, 'is_home': 1, 'vegas_total': vegas_tot, 'team_implied': round(h_imp, 1), 'opp_implied': round(a_imp, 1), 'team_margin': round(h_imp - a_imp, 1), 'matchup_str': f"vs {a_team}", 'is_dome': is_dome, 'wind': wind_spd, 'temp': temp_f, 'rest_days': h_rest, 'div_game': is_div, 'env_tag': env_tag}
        a_dict = {'opp': h_team, 'is_home': 0, 'vegas_total': vegas_tot, 'team_implied': round(a_imp, 1), 'opp_implied': round(h_imp, 1), 'team_margin': round(a_imp - h_imp, 1), 'matchup_str': f"@ {h_team}", 'is_dome': is_dome, 'wind': wind_spd, 'temp': temp_f, 'rest_days': a_rest, 'div_game': is_div, 'env_tag': env_tag}
        hist_env_map[(ssn, wk, h_team)] = h_dict
        hist_env_map[(ssn, wk, a_team)] = a_dict
        if ssn == 2026:
            schedule_map[(wk, h_team)] = h_dict
            schedule_map[(wk, a_team)] = a_dict
    STATE['schedule_map'] = schedule_map
    del sched_df, sched_full
    gc.collect()

    sleeper_pid_to_info, norm_to_bio, gsis_to_bio = {}, {}, {}
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
            resp = requests.get("https://api.sleeper.app/v1/players/nfl", headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            if resp.status_code == 200:
                for pid, pdata in resp.json().items():
                    if not isinstance(pdata, dict):
                        continue
                    pos = pdata.get('position', '')
                    if pos in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']:
                        fname = f"{pdata.get('first_name', '')} {pdata.get('last_name', '')}".strip()
                        t_abbr = team_fix.get(pdata.get('team') or '', pdata.get('team') or str(pid))
                        gsis = str(pdata.get('gsis_id') or '').strip()
                        exp = pdata.get('years_exp')
                        exp_str = "Rookie" if exp == 0 else (f"{exp} yrs" if exp is not None else "Vet")
                        headshot_url = f"https://sleepercdn.com/content/nfl/players/thumb/{pid}.jpg" if pos != 'DEF' else f"https://sleepercdn.com/images/team_logos/nfl/{str(t_abbr).lower()}.png"
                        bio_obj = {'sleeper_id': str(pid), 'name': fname if fname else str(pid), 'norm': norm_name(fname), 'pos': pos, 'team': t_abbr, 'gsis_id': gsis, 'college': pdata.get('college') or 'NFL', 'age': pdata.get('age') or 'N/A', 'exp': exp_str, 'height': pdata.get('height') or '', 'weight': pdata.get('weight') or '', 'headshot': headshot_url}
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
        except Exception:
            pass
    STATE['sleeper_pid_to_info'] = sleeper_pid_to_info
    STATE['norm_to_bio'] = norm_to_bio
    STATE['gsis_to_bio'] = gsis_to_bio
    gc.collect()

    cache_loaded = False
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cb = pickle.load(f)
            if cb.get('version') == 'v9_props':
                STATE['df'] = cb['df']
                STATE['latest_dvp'] = cb['latest_dvp']
                STATE['model_mean'] = cb['model_mean']
                STATE['model_median'] = cb['model_median']
                STATE['model_mae'] = cb['model_mae']
                STATE['model_floor'] = cb['model_floor']
                STATE['model_boom'] = cb['model_boom']
                STATE['prop_models'] = cb['prop_models']
                STATE['pos_baselines'] = cb['pos_baselines']
                STATE['name_col'] = cb['name_col']
                STATE['team_col'] = cb['team_col']
                STATE['opp_col'] = cb['opp_col']
                cache_loaded = True
                del cb
                gc.collect()
        except Exception:
            cache_loaded = False

    if not cache_loaded:
        dfs = []
        for y in [2024, 2025, 2026]:
            url = f"https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{y}.csv"
            try:
                dfs.append(pd.read_csv(url, low_memory=False))
            except Exception:
                pass
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()

        name_col = 'player_display_name' if 'player_display_name' in df.columns else 'player_name'
        team_col = 'team' if 'team' in df.columns else 'recent_team'
        opp_col = 'opponent_team' if 'opponent_team' in df.columns else 'opponent'
        STATE['name_col'], STATE['team_col'], STATE['opp_col'] = name_col, team_col, opp_col

        df[team_col] = df[team_col].replace(team_fix)
        df[opp_col] = df[opp_col].replace(team_fix)
        df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()

        stat_cols = ['fantasy_points_ppr', 'targets', 'receptions', 'receiving_yards', 'receiving_tds', 'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa', 'target_share', 'air_yards_share', 'wopr', 'racr', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_first_downs', 'rushing_epa', 'passing_yards', 'passing_tds', 'interceptions', 'passing_first_downs', 'passing_epa', 'completions', 'attempts']
        for c in stat_cols:
            df[c] = df[c].fillna(0.0) if c in df.columns else 0.0

        df = df.sort_values(['player_id', 'season', 'week']).reset_index(drop=True)
        vegas_tot_list, team_imp_list, team_mar_list, dome_list, wind_list, rest_list, div_list = [], [], [], [], [], [], []
        for _, r in df.iterrows():
            env = hist_env_map.get((int(r['season']), int(r['week']), r[team_col]))
            if env:
                vegas_tot_list.append(env['vegas_total']); team_imp_list.append(env['team_implied']); team_mar_list.append(env['team_margin']); dome_list.append(env['is_dome']); wind_list.append(env['wind']); rest_list.append(env['rest_days']); div_list.append(env['div_game'])
            else:
                vegas_tot_list.append(44.5); team_imp_list.append(22.2); team_mar_list.append(0.0); dome_list.append(0); wind_list.append(6.0); rest_list.append(7); div_list.append(0)

        df['vegas_total'] = vegas_tot_list; df['team_implied'] = team_imp_list; df['team_margin'] = team_mar_list; df['is_dome'] = dome_list; df['wind_speed'] = wind_list; df['rest_days'] = rest_list; df['div_game'] = div_list
        df['calc_wopr'] = np.where(df['wopr'] > 0, df['wopr'], 1.5 * df['target_share'] + 0.7 * df['air_yards_share'])
        df['calc_racr'] = np.where(df['racr'] > 0, df['racr'], np.where(df['receiving_air_yards'] >= 10, df['receiving_yards'] / np.maximum(1.0, df['receiving_air_yards']), 1.0))
        df['total_epa'] = df['passing_epa'] + df['rushing_epa'] + df['receiving_epa']
        df['total_first_downs'] = df['passing_first_downs'] + df['rushing_first_downs'] + df['receiving_first_downs']
        df['td_equity'] = df['rushing_tds'] + df['receiving_tds'] + df['passing_tds'] * 0.5
        df['non_qb_tds'] = df['rushing_tds'] + df['receiving_tds']

        ypc_edge = np.where(df['carries'] >= 3, (df['rushing_yards'] / np.maximum(1, df['carries'])) - 4.3, 0.0)
        ypt_edge = np.where(df['targets'] >= 2, (df['receiving_yards'] / np.maximum(1, df['targets'])) - 7.8, 0.0)
        df['eff_edge'] = np.clip(ypc_edge * 0.45 + ypt_edge * 0.35, -3.5, 4.5)
        df['xFP_volume'] = df['targets'] * 1.25 + df['receptions'] * 0.38 + df['receiving_air_yards'] * 0.042 + df['receiving_first_downs'] * 0.55 + df['carries'] * 0.74 + df['rushing_yards'] * 0.048 + df['rushing_first_downs'] * 0.52 + df['passing_yards'] * 0.04 + df['passing_tds'] * 3.60 + (df['rushing_tds'] + df['receiving_tds']) * 2.15 - df['interceptions'] * 1.50
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
        df['roll3_eff_edge'] = g['eff_edge'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
        df['roll3_td_equity'] = g['td_equity'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0.0)
        df['talent_prior'] = 0.36 * df['season_ppg_prior'] + 0.28 * df['ewma_pts'] + 0.26 * df['roll3_xFP'] + 0.10 * df['career_ppg_prior']

        def_pos = df.groupby(['season', 'week', opp_col, 'position'])['fantasy_points_ppr'].sum().reset_index().sort_values(['season', 'week'])
        def_pos['opp_pts_allowed'] = def_pos.groupby([opp_col, 'position'])['fantasy_points_ppr'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        pos_avg = def_pos.groupby('position')['fantasy_points_ppr'].transform('mean')
        def_pos['matchup_boost'] = ((def_pos['opp_pts_allowed'] - pos_avg) * 0.48).fillna(0.0)
        df = df.merge(def_pos[['season', 'week', opp_col, 'position', 'matchup_boost']], on=['season', 'week', opp_col, 'position'], how='left')
        df['matchup_boost'] = df['matchup_boost'].fillna(0.0)
        STATE['latest_dvp'] = def_pos[def_pos['season'] == 2026].sort_values('week').groupby([opp_col, 'position'])['matchup_boost'].last().to_dict()
        df['pos_code'] = df['position'].map(pos_map)

        train_df = df.dropna(subset=features + ['fantasy_points_ppr']).copy()
        train_df = train_df[train_df['fantasy_points_ppr'] >= 0.0]
        X_train, y_train = train_df[features], train_df['fantasy_points_ppr']
        STATE['model_mean'] = HistGradientBoostingRegressor(loss='squared_error', max_iter=200, max_depth=5, learning_rate=0.045, l2_regularization=0.20, random_state=42).fit(X_train, y_train)
        STATE['model_mae'] = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, max_depth=5, learning_rate=0.045, l2_regularization=0.20, random_state=42).fit(X_train, y_train)
        STATE['model_median'] = HistGradientBoostingRegressor(loss='quantile', quantile=0.50, max_iter=180, max_depth=5, learning_rate=0.045, l2_regularization=0.20, random_state=42).fit(X_train, y_train)
        STATE['model_floor'] = HistGradientBoostingRegressor(loss='quantile', quantile=0.10, max_iter=130, max_depth=4, learning_rate=0.05, random_state=42).fit(X_train, y_train)
        STATE['model_boom'] = HistGradientBoostingRegressor(loss='quantile', quantile=0.90, max_iter=130, max_depth=4, learning_rate=0.05, random_state=42).fit(X_train, y_train)

        prop_models = {}
        for prop_key, target_col in [('pass_yd', 'passing_yards'), ('rush_yd', 'rushing_yards'), ('rec_yd', 'receiving_yards'), ('receptions', 'receptions'), ('any_td', 'non_qb_tds')]:
            y_prop = train_df[target_col].clip(lower=0.0)
            if prop_key == 'any_td':
                prop_models[prop_key] = ('mean', HistGradientBoostingRegressor(loss='squared_error', max_iter=150, max_depth=4, learning_rate=0.045, random_state=42).fit(X_train, y_prop))
            else:
                m_med = HistGradientBoostingRegressor(loss='quantile', quantile=0.50, max_iter=160, max_depth=5, learning_rate=0.045, random_state=42).fit(X_train, y_prop)
                m_mean = HistGradientBoostingRegressor(loss='squared_error', max_iter=160, max_depth=5, learning_rate=0.045, random_state=42).fit(X_train, y_prop)
                prop_models[prop_key] = ('blend', m_med, m_mean)
        STATE['prop_models'] = prop_models

        starter_ref = train_df[train_df['talent_prior'] >= 7.0]
        STATE['pos_baselines'] = {p_code: (starter_ref[starter_ref['pos_code'] == p_code] if not starter_ref[starter_ref['pos_code'] == p_code].empty else train_df[train_df['pos_code'] == p_code])[features].median().to_dict() for p_code in [1, 2, 3, 4]}
        STATE['df'] = df
        del train_df
        gc.collect()

    df = STATE['df']
    name_col, team_col = STATE['name_col'], STATE['team_col']
    df_2026 = df[df['season'] == 2026].copy()
    STATE['df_2026'] = df_2026
    STATE['latest_completed_week'] = int(df_2026['week'].max())
    STATE['default_target_week'] = min(STATE['latest_completed_week'] + 1, 18)
    STATE['active_player_names'] = sorted(df_2026[name_col].dropna().unique().tolist())

    playoff_sos_map = {}
    for t_abbr in df_2026[team_col].dropna().unique().tolist():
        for pos in ['QB', 'RB', 'WR', 'TE']:
            po_boosts = [float(STATE['latest_dvp'].get((STATE['schedule_map'][(po_wk, t_abbr)]['opp'], pos), 0.0)) for po_wk in [15, 16, 17] if (po_wk, t_abbr) in STATE['schedule_map']]
            playoff_sos_map[(t_abbr, pos)] = round(float(np.mean(po_boosts)) if po_boosts else 0.0, 2)
    STATE['playoff_sos_map'] = playoff_sos_map
    STATE['ready'] = True
    gc.collect()

def get_player_bio(p_name, p_pos, p_team, gsis_id=""):
    norm_k = norm_name(p_name)
    bio = STATE['gsis_to_bio'].get(str(gsis_id)) or STATE['norm_to_bio'].get((norm_k, p_pos)) or STATE['norm_to_bio'].get(norm_k)
    fallback_logo = f"https://sleepercdn.com/images/team_logos/nfl/{str(p_team).lower()}.png"
    if bio:
        return {'headshot': bio.get('headshot') or fallback_logo, 'college': bio.get('college') or 'NFL', 'age': bio.get('age') or 'N/A', 'exp': bio.get('exp') or 'Vet', 'height': bio.get('height') or '', 'weight': bio.get('weight') or ''}
    return {'headshot': fallback_logo, 'college': 'NFL', 'age': 'N/A', 'exp': 'Active', 'height': '', 'weight': ''}

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
                inj = p_info.get('injury_status')
                p_stat = p_info.get('status')
                key = (norm_name(full_name), pos)
                if inj in ['Out', 'IR', 'PUP', 'Doubtful', 'Sus']: status_lookup[key] = inj.upper()
                elif p_stat in ['Inactive', 'Injured Reserve']: status_lookup[key] = 'OUT'
                elif inj == 'Questionable': status_lookup[key] = 'Q'
                else: status_lookup[key] = 'ACT'
                if pts > 0:
                    proj_lookup[key] = round(pts, 2)
                    if pid: pid_proj_lookup[pid] = round(pts, 2)
                    stat_proj_lookup[key] = {'pass_yd': round(float(stats.get('pass_yd', 0.0) or 0.0), 1), 'rush_yd': round(float(stats.get('rush_yd', 0.0) or 0.0), 1), 'rec_yd': round(float(stats.get('rec_yd', 0.0) or 0.0), 1), 'receptions': round(float(stats.get('rec', 0.0) or 0.0), 1), 'any_td': round(float((stats.get('rush_td', 0.0) or 0.0) + (stats.get('rec_td', 0.0) or 0.0)), 2)}
                elif status_lookup[key] in ['OUT', 'IR', 'PUP', 'DOUBTFUL']:
                    proj_lookup[key] = 0.0
                    if pid: pid_proj_lookup[pid] = 0.0
    except Exception:
        pass
    sleeper_cache[week_num] = (proj_lookup, status_lookup, pid_proj_lookup, stat_proj_lookup)
    return proj_lookup, status_lookup, pid_proj_lookup, stat_proj_lookup

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

def prob_to_american_odds(prob_pct):
    p = np.clip(float(prob_pct) / 100.0, 0.02, 0.98)
    if p >= 0.50: return f"{-int(round((p / (1.0 - p)) * 100.0))}"
    return f"+{int(round(((1.0 - p) / p) * 100.0))}"

def american_odds_to_implied_prob(odds_val):
    try: o = float(odds_val)
    except Exception: o = -110.0
    if o < 0: return (-o) / (-o + 100.0)
    elif o > 0: return 100.0 / (o + 100.0)
    return 0.5238

def render_payoff_distribution_bar_html(p_name, pos_team, flr_val, mod_val, slp_val, ceil_val, max_scale=34.0, compact=False):
    flr_pct = np.clip((flr_val / max_scale) * 100.0, 1.0, 96.0)
    mod_pct = np.clip((mod_val / max_scale) * 100.0, flr_pct + 1.0, 98.0)
    slp_pct = np.clip((slp_val / max_scale) * 100.0, 1.0, 98.0)
    ceil_pct = np.clip((ceil_val / max_scale) * 100.0, mod_pct + 1.5, 99.5)
    span_width = max(2.5, ceil_pct - flr_pct)
    pad = "8px 12px" if compact else "12px 16px"
    return (
        f'<div style="padding:{pad}; border-radius:10px; border:1px solid rgba(148,163,184,0.3); background:rgba(15,23,42,0.02); margin-bottom:8px;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; font-size:0.86rem; margin-bottom:6px;">'
        f'<div><strong>{p_name}</strong> <span style="opacity:0.75; font-size:0.8rem;">({pos_team})</span></div>'
        f'<div style="font-size:0.82rem;"><span style="opacity:0.75;">Floor: <strong>{flr_val:.1f}</strong></span> &nbsp;|&nbsp; <span style="color:#2563eb; font-weight:700;">Model: {mod_val:.2f}</span> &nbsp;|&nbsp; <span style="opacity:0.8;">Sleeper: {slp_val:.2f}</span> &nbsp;|&nbsp; <span style="color:#059669; font-weight:700;">Ceiling: {ceil_val:.1f}</span></div></div>'
        f'<div style="position:relative; height:18px; border-radius:9px; background:rgba(148,163,184,0.18); overflow:hidden;">'
        f'<div style="position:absolute; left:29.4%; top:0; bottom:0; width:1px; background:rgba(148,163,184,0.35);"></div>'
        f'<div style="position:absolute; left:58.8%; top:0; bottom:0; width:1px; background:rgba(148,163,184,0.35);"></div>'
        f'<div style="position:absolute; left:88.2%; top:0; bottom:0; width:1px; background:rgba(148,163,184,0.35);"></div>'
        f'<div style="position:absolute; left:{flr_pct:.1f}%; width:{span_width:.1f}%; top:3px; bottom:3px; border-radius:6px; background:linear-gradient(90deg, rgba(59,130,246,0.35) 0%, rgba(37,99,235,0.65) 55%, rgba(16,185,129,0.75) 100%);"></div>'
        f'<div style="position:absolute; left:calc({slp_pct:.1f}% - 2px); top:1px; bottom:1px; width:4px; border-radius:2px; background:#94a3b8; border:1px solid #475569;"></div>'
        f'<div style="position:absolute; left:calc({mod_pct:.1f}% - 5px); top:1px; width:10px; height:16px; border-radius:4px; background:#1d4ed8; border:2px solid #ffffff; box-shadow:0 1px 3px rgba(0,0,0,0.3);"></div></div>'
        f'<div style="display:flex; justify-content:space-between; font-size:0.7rem; opacity:0.6; margin-top:3px;"><span>0 PPR</span><span>10 PPR</span><span>20 PPR</span><span>30+ PPR</span></div></div>'
    )

def compute_shapley_group_attribution(p_row):
    p_code = int(p_row['pos_code'])
    base_vec = dict(STATE['pos_baselines'].get(p_code, {f: 0.0 for f in features}))
    def eval_ens(vec_dict):
        arr = pd.DataFrame([vec_dict])[features]
        return float(0.42 * STATE['model_mean'].predict(arr)[0] + 0.33 * STATE['model_mae'].predict(arr)[0] + 0.25 * STATE['model_median'].predict(arr)[0])
    base_pred = eval_ens(base_vec)
    vol_feats = ['talent_prior', 'season_ppg_prior', 'career_ppg_prior', 'ewma_pts', 'roll2_pts', 'roll3_pts', 'roll5_pts', 'roll3_xFP', 'roll2_xFP', 'season_xFP_prior', 'role_trend', 'roll3_targets', 'roll3_receptions', 'roll3_carries', 'roll3_pass_yds', 'roll3_td_equity']
    eff_feats = ['roll3_tgt_share', 'roll3_air_share', 'roll3_wopr', 'roll3_racr', 'roll3_epa', 'roll3_first_downs', 'roll3_air_yards', 'roll3_yac', 'roll3_eff_edge', 'roll3_luck', 'roll5_std']
    v1 = dict(base_vec)
    for f in vol_feats: v1[f] = p_row[f]
    p1 = eval_ens(v1)
    v2 = dict(v1)
    for f in eff_feats: v2[f] = p_row[f]
    p2 = eval_ens(v2)
    v3 = dict(v2)
    v3['matchup_boost'] = p_row['matchup_boost']
    p3 = eval_ens(v3)
    v4 = dict(v3)
    for f in ['vegas_total', 'team_implied', 'team_margin', 'is_dome', 'wind_speed', 'rest_days', 'div_game']: v4[f] = p_row[f]
    p4 = eval_ens(v4)
    return {
        'Positional Starter Baseline': round(base_pred, 2),
        'Volume & Role Equity (xFP / Touches)': round(p1 - base_pred, 2),
        'Play-by-Play Efficiency (WOPR / EPA / 1st Downs)': round(p2 - p1, 2),
        'Defense vs. Position (Normalized DvP)': round((p3 - p2) * 0.72 + float(p_row['matchup_boost']) * 0.14, 2),
        'Vegas Script, Weather & Vacated Volume': round((p4 - p3) * 0.72 + (float(p_row['team_implied']) - 22.0) * 0.12 + float(p_row['Vacated Boost']) * 0.45, 2)
    }

def build_week_projections(target_week, force_refresh=False):
    ensure_engine_initialized()
    target_week = int(target_week)
    if not force_refresh and target_week in projection_week_cache:
        return projection_week_cache[target_week].copy()

    df = STATE['df']; df_2026 = STATE['df_2026']; name_col, team_col = STATE['name_col'], STATE['team_col']
    latest_completed_week = STATE['latest_completed_week']; is_past_week = (target_week <= latest_completed_week)
    sleeper_map, injury_map, _, stat_proj_map = get_sleeper_week_data(target_week, force_refresh=force_refresh)

    cutoff_df = df_2026[df_2026['week'] < target_week].copy() if is_past_week and target_week > 1 else df_2026.copy()
    if cutoff_df.empty: cutoff_df = df[df['season'] == 2025].copy()

    roster_pool = df_2026.sort_values('week').groupby('player_id').last().reset_index()
    actual_map = dict(zip(df_2026[df_2026['week'] == target_week]['player_id'], df_2026[df_2026['week'] == target_week]['fantasy_points_ppr'].round(2))) if is_past_week else {}

    rows = []
    for _, player in roster_pool.iterrows():
        pid, p_name, p_pos, p_team = str(player['player_id']).strip(), player[name_col], player['position'], player[team_col]
        p_hist = cutoff_df[cutoff_df['player_id'] == player['player_id']].sort_values(['season', 'week'])
        if p_hist.empty: continue
        full_2026_hist = df_2026[df_2026['player_id'] == player['player_id']].sort_values('week')
        full_career_hist = df[df['player_id'] == player['player_id']].sort_values(['season', 'week'])

        sched_info = STATE['schedule_map'].get((target_week, p_team), None)
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
        if pd.isna(r5_std) or r5_std < 2.0: r5_std = 4.0

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
        m_boost = 0.0 if is_bye else (0.0 if is_out else float(STATE['latest_dvp'].get((sched_info['opp'], p_pos), 0.0)))
        vegas_tot = 44.5 if is_bye else float(sched_info['vegas_total'])
        team_imp = 22.0 if is_bye else float(sched_info['team_implied'])
        team_margin = 0.0 if is_bye else float(sched_info['team_margin'])
        is_dome = 0 if is_bye else int(sched_info['is_dome'])
        wind_spd = 6.0 if is_bye else float(sched_info['wind'])
        rest_days = 7 if is_bye else int(sched_info['rest_days'])
        div_game = 0 if is_bye else int(sched_info['div_game'])
        env_tag = "Bye Week" if is_bye else sched_info['env_tag']

        status_badge = "BYE" if is_bye else (f"{inj_code if inj_code != 'ACT' else 'OUT'}" if is_out else ("QUESTIONABLE" if is_q else "ACTIVE"))
        po_sos = STATE['playoff_sos_map'].get((p_team, p_pos), 0.0)

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
            vac_boost_list.append(0.0); continue
        t_abbr = r['Team']
        v_tgt = vacated_tgts_by_team.get(t_abbr, 0.0); v_car = vacated_cars_by_team.get(t_abbr, 0.0)
        v_boost = 0.0
        if r['Pos'] in ['WR', 'TE'] and v_tgt >= 5.0 and r['roll3_targets'] >= 3.0:
            v_boost += min(0.95, v_tgt * 0.07 * min(1.4, max(0.5, r['roll3_tgt_share'] * 4.0)))
        elif r['Pos'] == 'RB' and v_car >= 7.0 and r['roll3_carries'] >= 4.0:
            v_boost += min(1.15, v_car * 0.09)
        vac_boost_list.append(round(float(v_boost), 2))
    wk_df['Vacated Boost'] = vac_boost_list

    X_wk = wk_df[features]
    raw_ml = 0.42 * STATE['model_mean'].predict(X_wk) + 0.33 * STATE['model_mae'].predict(X_wk) + 0.25 * STATE['model_median'].predict(X_wk)
    implied_adj = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, (wk_df['team_implied'] - 22.0) * 0.18 + wk_df['Vacated Boost'] * 0.45)
    epa_fd_adj = np.clip(wk_df['roll3_epa'] * 0.06 + (wk_df['roll3_first_downs'] - 2.5) * 0.11, -0.9, 1.1)
    pure_model_proj = 0.72 * raw_ml + 0.28 * (wk_df['talent_prior'] + wk_df['matchup_boost'] * 0.50 + implied_adj + epa_fd_adj)

    slp_list = []
    for _, r in wk_df.iterrows():
        if r['is_bye'] or r['is_out']: slp_list.append(0.0)
        else:
            slp_val = sleeper_map.get((r['norm_name'], r['Pos']), None)
            slp_list.append(slp_val if slp_val is not None else round(max(0.0, 0.65 * r['season_ppg_prior'] + 0.35 * r['roll3_pts'] + r['matchup_boost'] * 0.3), 2))
    wk_df[COL_SLEEPER] = slp_list

    w_ml = STATE['optimal_ml_weight']
    market_anchored = np.where(wk_df[COL_SLEEPER] > 0, w_ml * pure_model_proj + (1.0 - w_ml) * wk_df[COL_SLEEPER], pure_model_proj)
    wk_df[COL_MODEL] = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, np.clip(market_anchored, 0, None)).round(2)
    wk_df[COL_DIFF] = (wk_df[COL_MODEL] - wk_df[COL_SLEEPER]).round(2)
    wk_df['Floor (10%)'] = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, np.clip(STATE['model_floor'].predict(X_wk), 0, wk_df[COL_MODEL] * 0.82)).round(2)
    wk_df['Boom (90%)'] = np.where(wk_df['is_bye'] | wk_df['is_out'], 0.0, np.maximum(STATE['model_boom'].predict(X_wk) + implied_adj, wk_df[COL_MODEL] + 3.4)).round(2)
    wk_df['ROS Value'] = np.where(wk_df['Status'].isin(['IR', 'PUP']), wk_df['season_healthy_ppg'] * 0.40, np.clip(wk_df['season_healthy_ppg'] + 0.15 * wk_df['Playoff SoS'], 0.0, None)).round(2)

    active_mask = ~(wk_df['is_bye'] | wk_df['is_out'])
    for p_key, roll_col, s_col, dvp_scale in [('pass_yd', 'roll3_pass_yds', 's_pyds', 4.5), ('rush_yd', 'roll3_rush_yds', 's_ruyds', 2.2), ('rec_yd', 'roll3_rec_yds', 's_reyds', 2.4), ('receptions', 'roll3_receptions', 's_rec', 0.18)]:
        _, m_med, m_mean = STATE['prop_models'][p_key]
        raw_p = 0.62 * m_med.predict(X_wk) + 0.38 * m_mean.predict(X_wk)
        hist_base = 0.55 * wk_df[roll_col] + 0.45 * wk_df[s_col] + wk_df['matchup_boost'] * dvp_scale
        pure_stat = np.clip(0.68 * raw_p + 0.32 * hist_base, 0.0, None)
        mkt_lines = []
        for idx, r in wk_df.iterrows():
            if not active_mask.iloc[idx]: mkt_lines.append(0.0); continue
            m_val = stat_proj_map.get((r['norm_name'], r['Pos']), {}).get(p_key, 0.0)
            if m_val <= 0.0: m_val = max(0.0, round(hist_base.iloc[idx], 1))
            mkt_lines.append(math.floor(m_val) + 0.5 if m_val >= 1.0 else 0.0)
        wk_df[f'Mkt_{p_key}'] = mkt_lines
        blended_stat = np.where(wk_df[f'Mkt_{p_key}'] > 0, 0.64 * pure_stat + 0.36 * wk_df[f'Mkt_{p_key}'], pure_stat)
        wk_df[f'Proj_{p_key}'] = np.where(active_mask, np.clip(blended_stat, 0.0, None), 0.0).round(1)

    _, m_td = STATE['prop_models']['any_td']
    raw_td_lambda = np.clip(0.65 * m_td.predict(X_wk) + 0.35 * (wk_df['roll3_td_equity'] * 0.85 + np.maximum(0.0, wk_df['team_implied'] - 20.0) * 0.03), 0.02, 1.85)
    wk_df['Anytime TD %'] = np.where(active_mask & (wk_df['Pos'].isin(['RB', 'WR', 'TE'])), (1.0 - np.exp(-raw_td_lambda)) * 100.0, 0.0).round(1)

    boom_pct_list, bust_pct_list, boom_reasons, bust_reasons, diff_reasons = [], [], [], [], []
    for _, r in wk_df.iterrows():
        proj = float(r[COL_MODEL])
        if r['is_bye'] or r['is_out'] or proj <= 1.0:
            boom_pct_list.append(0.0); bust_pct_list.append(100.0 if r['is_out'] else 0.0); boom_reasons.append("Inactive / Bye"); bust_reasons.append("Inactive / Bye"); diff_reasons.append("Inactive / Bye"); continue
        xfp_edge = float(r['roll3_xFP'] - r['roll3_pts']); dvp = float(r['matchup_boost']); vegas_edge = float(r['team_implied'] - 22.0)
        model_edge = float(r[COL_DIFF]); wopr_val = float(r['roll3_wopr']); epa_val = float(r['roll3_epa']); vac_val = float(r['Vacated Boost']); vol_ratio = float(r['roll5_std']) / max(proj, 7.0)
        skew = (0.34 * xfp_edge) + (0.31 * dvp) + (0.18 * model_edge) + (0.10 * vegas_edge) + (0.08 * epa_val) + (0.20 * vac_val)
        base_rate = np.clip(16.0 + vol_ratio * 6.0, 15.0, 21.0)
        p_boom = base_rate + (2.7 * skew) + (1.3 * max(0.0, dvp)) + (0.85 * max(0.0, xfp_edge))
        p_bust = base_rate - (2.7 * skew) + (1.4 * max(0.0, -dvp)) + (0.95 * max(0.0, -xfp_edge)) + (7.5 if r['is_q'] else 0.0)
        boom_pct_list.append(round(float(np.clip(p_boom, 5.0, 47.0)), 1))
        bust_pct_list.append(round(float(np.clip(p_bust, 5.0, 48.0)), 1))
        b_drivers = []
        if vac_val >= 0.5: b_drivers.append(f"Vacated Teammate Vol (+{vac_val:.1f})")
        if xfp_edge >= 1.4: b_drivers.append(f"Unconverted Volume ({r['roll3_xFP']:.1f} xFP)")
        if wopr_val >= 0.55 and r['Pos'] in ['WR', 'TE']: b_drivers.append(f"Alpha WOPR ({wopr_val:.2f})")
        if dvp >= 1.2: b_drivers.append(f"Soft Defense ({dvp:+.1f})")
        if not b_drivers: b_drivers.append(f"High Ceiling ({r['Boom (90%)']:.1f})")
        boom_reasons.append(" + ".join(b_drivers[:2]))
        d_drivers = []
        if xfp_edge <= -1.4: d_drivers.append(f"Overperformed Volume")
        if dvp <= -1.2: d_drivers.append(f"Tough Defense ({dvp:+.1f})")
        if r['is_q']: d_drivers.append("Questionable Tag")
        if not d_drivers: d_drivers.append(f"Fragile Floor ({r['Floor (10%)']:.1f})")
        bust_reasons.append(" + ".join(d_drivers[:2]))
        diff_reasons.append(f"Trailing {r['roll3_xFP']:.1f} xFP + {dvp:+.1f} DvP" if model_edge >= 0 else f"Consensus inflated")

    wk_df['Boom %'] = boom_pct_list; wk_df['Bust %'] = bust_pct_list; wk_df['Boom Catalyst'] = boom_reasons; wk_df['Bust Catalyst'] = bust_reasons; wk_df['Discrepancy Reason'] = diff_reasons
    wk_df['Outlook Verdict'] = wk_df.apply(lambda r: "ON BYE WEEK" if r['is_bye'] else (f"INACTIVE ({r['Status']})" if r['is_out'] else ("QUESTIONABLE" if r['is_q'] else ("HIGH-CEILING SKEW" if r['Boom %'] >= 26.5 and r['Boom %'] - r['Bust %'] >= 7.0 else ("HIGH BUST RISK" if r['Bust %'] >= 26.5 and r['Bust %'] - r['Boom %'] >= 7.0 else "EXPECTED BASELINE")))), axis=1)
    wk_df['L3 Avg PPR'] = wk_df['roll3_pts'].round(2); wk_df['L3 xFP (Vol)'] = wk_df['roll3_xFP'].round(2); wk_df['WOPR'] = wk_df['roll3_wopr'].round(2); wk_df['RACR'] = wk_df['roll3_racr'].round(2); wk_df['L3 EPA'] = wk_df['roll3_epa'].round(2); wk_df['DvP Boost'] = wk_df['matchup_boost'].round(2)
    sorted_df = wk_df.sort_values(by=COL_MODEL, ascending=False).reset_index(drop=True)
    projection_week_cache[target_week] = sorted_df
    return sorted_df.copy()

def update_main_table(week_num, hide_byes, hide_injured, compact_view, pos_filter, outlook_filter, search_text, force_refresh=False):
    wk_df = build_week_projections(int(week_num), force_refresh=force_refresh)
    our_20plus = int((wk_df[COL_MODEL] >= 20.0).sum()); slp_20plus = int((wk_df[COL_SLEEPER] >= 20.0).sum())
    mode_label = "Completed Week Backtest" if int(week_num) <= STATE['latest_completed_week'] else "Live Real-Time Projection"
    banner = f"**Week {int(week_num)} Overview ({mode_label})** — Projected 20+ Scorers: Model: **{our_20plus}** vs. Sleeper: **{slp_20plus}** | *Click any row for Visual Payoff Distribution & Feature Waterfall.*"
    if hide_byes: wk_df = wk_df[~wk_df['is_bye']].copy()
    if hide_injured: wk_df = wk_df[~wk_df['is_out']].copy()
    if pos_filter != "ALL": wk_df = wk_df[wk_df['Pos'] == pos_filter].copy()
    if outlook_filter != "ALL": wk_df = wk_df[wk_df['Outlook Verdict'].str.contains(outlook_filter, case=False, na=False)].copy()
    if search_text: wk_df = wk_df[wk_df['Player'].str.contains(search_text, case=False, na=False) | wk_df['Team'].str.contains(search_text, case=False, na=False)].copy()
    cols = ['Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Floor (10%)', 'Boom (90%)', 'Boom %', 'Bust %', 'ROS Value', 'Outlook Verdict'] if compact_view else ['Player', 'Pos', 'Team', 'Status', 'Matchup', 'Env', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Floor (10%)', 'Boom (90%)', 'Boom %', 'Bust %', 'ROS Value', 'L3 Avg PPR', 'L3 xFP (Vol)', 'Vacated Boost', 'WOPR', 'RACR', 'L3 EPA', 'DvP Boost', 'Playoff SoS', 'Outlook Verdict']
    if int(week_num) <= STATE['latest_completed_week']: cols.insert(7 if compact_view else 9, 'Actual PPR')
    return banner, wk_df[cols].reset_index(drop=True)

def force_refresh_main_table(week_num, hide_byes, hide_injured, compact_view, pos_filter, outlook_filter, search_text):
    return update_main_table(week_num, hide_byes, hide_injured, compact_view, pos_filter, outlook_filter, search_text, force_refresh=True)

def open_player_card_on_click(table_df, week_num, evt: gr.SelectData):
    if table_df is None or len(table_df) == 0: return gr.update(visible=False), "", gr.update(visible=False)
    p_name = str(table_df.iloc[evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index]['Player'])
    wk_df = build_week_projections(int(week_num))
    p_rows = wk_df[wk_df['Player'] == p_name]
    if p_rows.empty: return gr.update(visible=False), "", gr.update(visible=False)
    p = p_rows.iloc[0]; bio = get_player_bio(p['Player'], p['Pos'], p['Team'], p.get('player_id', ''))
    dist_bar_html = render_payoff_distribution_bar_html(f"{p['Player']} Payoff Distribution", f"{p['Pos']} - {p['Team']}", float(p['Floor (10%)']), float(p[COL_MODEL]), float(p[COL_SLEEPER]), float(p['Boom (90%)']))
    shap_dict = compute_shapley_group_attribution(p)
    shap_items = "".join([f'<div style="padding:8px; border:1px solid rgba(148,163,184,0.25); border-radius:6px;"><div style="font-size:0.75rem;">{k}</div><div style="font-weight:700; color:{"#475569" if "Baseline" in k else ("#059669" if v >= 0 else "#dc2626")};">{f"{v:.2f} PPR" if "Baseline" in k else f"{v:+.2f} PPR"}</div></div>' for k, v in shap_dict.items()])
    card_html = f'<div style="padding:16px; border:1px solid rgba(148,163,184,0.35); border-radius:10px; background:rgba(15,23,42,0.03); margin-bottom:12px;"><div style="display:flex; gap:16px; align-items:center; margin-bottom:12px;"><img src="{bio["headshot"]}" style="width:80px; height:80px; border-radius:50%; object-fit:cover; background:#1e293b;" onerror="this.src='https://sleepercdn.com/images/v2/icons/player_default.webp'"/><div><div style="font-size:1.2rem; font-weight:700;">{p["Player"]} — {p["Outlook Verdict"]}</div><div style="font-size:0.85rem; opacity:0.8;">{p["Pos"]} — {p["Team"]} | Age: {bio["age"]} | Exp: {bio["exp"]} | College: {bio["college"]}</div></div></div>{dist_bar_html}<div style="font-weight:700; font-size:0.85rem; margin:8px 0 4px 0;">Feature Attribution Waterfall:</div><div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:8px;">{shap_items}</div></div>'
    logs = STATE['df_2026'][STATE['df_2026']['player_id'].astype(str) == str(p['player_id'])].sort_values('week', ascending=False)
    logs_clean = logs[['week', STATE['opp_col'], 'fantasy_points_ppr', 'xFP_volume', 'calc_wopr', 'calc_racr', 'total_epa', 'targets', 'receptions', 'receiving_yards', 'carries', 'rushing_yards']].copy()
    for c in ['fantasy_points_ppr', 'xFP_volume', 'calc_wopr', 'calc_racr', 'total_epa']: logs_clean[c] = logs_clean[c].round(2)
    logs_clean.columns = ['Week', 'Opp', 'Actual PPR', 'Expected xFP', 'WOPR', 'RACR', 'Total EPA', 'Targets', 'Rec', 'Rec Yds', 'Carries', 'Rush Yds']
    return gr.update(visible=True), card_html, gr.update(value=logs_clean, visible=True)

def close_player_scouting_drawer(): return gr.update(visible=False), "", gr.update(visible=False)

def get_top_outliers(week_num):
    wk_df = build_week_projections(int(week_num))
    v = wk_df[((wk_df[COL_MODEL] >= 10.0) | (wk_df[COL_SLEEPER] >= 10.0)) & (~wk_df['is_out']) & (~wk_df['is_bye'])].copy()
    return v[v['Boom %'] > v['Bust %']].sort_values('Boom %', ascending=False).head(5)[['Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Boom (90%)', 'Boom %', 'Bust %', 'Boom Catalyst']].reset_index(drop=True), v[(v['Bust %'] > v['Boom %'])].sort_values('Bust %', ascending=False).head(5)[['Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Floor (10%)', 'Bust %', 'Boom %', 'Bust Catalyst']].reset_index(drop=True)

def get_biggest_differers(week_num, pos_filter):
    wk_df = build_week_projections(int(week_num))
    v = wk_df[(~wk_df['is_out']) & (~wk_df['is_bye']) & ((wk_df[COL_MODEL] >= 6.0) | (wk_df[COL_SLEEPER] >= 6.0))].copy()
    if pos_filter != "ALL": v = v[v['Pos'] == pos_filter].copy()
    cols = ['Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'L3 xFP (Vol)', 'Vacated Boost', 'WOPR', 'DvP Boost', 'Discrepancy Reason']
    return v[v[COL_DIFF] > 0].sort_values(by=COL_DIFF, ascending=False).head(12)[cols].reset_index(drop=True), v[v[COL_DIFF] < 0].sort_values(by=COL_DIFF, ascending=True).head(12)[cols].reset_index(drop=True)

def compare_multiple_players(week_num, player_list):
    if not player_list or len(player_list) < 2: return "", "Select at least 2 players.", pd.DataFrame()
    wk_df = build_week_projections(int(week_num))
    selected = [wk_df[wk_df['Player'] == p].iloc[0] for p in player_list[:6] if not wk_df[wk_df['Player'] == p].empty]
    if len(selected) < 2: return "", "Players not found.", pd.DataFrame()
    max_ceil = max([float(r['Boom (90%)']) for r in selected] + [32.0])
    bars = '<div style="padding:14px; border:1px solid rgba(148,163,184,0.35); border-radius:10px; margin-bottom:12px;"><div style="font-weight:700; margin-bottom:8px;">Synchronized Payoff Distribution Comparison</div>'
    for r in selected: bars += render_payoff_distribution_bar_html(r['Player'], f"{r['Pos']} - {r['Team']}", float(r['Floor (10%)']), float(r[COL_MODEL]), float(r[COL_SLEEPER]), float(r['Boom (90%)']), max_scale=max_ceil + 2.0, compact=True)
    bars += '</div>'
    td = {'Metric': ['Position / Team', 'Matchup', "Model's Projection", "Sleeper's Projection", "Diff", 'Floor', 'Ceiling', 'Boom %', 'Bust %', 'ROS Value']}
    for r in selected: td[r['Player']] = [f"{r['Pos']} ({r['Team']})", r['Matchup'], f"{r[COL_MODEL]:.2f}", f"{r[COL_SLEEPER]:.2f}", f"{r[COL_DIFF]:+.2f}", f"{r['Floor (10%)']:.2f}", f"{r['Boom (90%)']:.2f}", f"{r['Boom %']}%", f"{r['Bust %']}%", f"{r['ROS Value']:.2f}"]
    return bars, f"### Multi-Player Comparison (Week {int(week_num)})", pd.DataFrame(td)

def build_player_props_board(week_num, prop_category, pos_filter, min_edge_filter, search_txt):
    wk_df = build_week_projections(int(week_num))
    active = wk_df[(~wk_df['is_bye']) & (~wk_df['is_out']) & (wk_df[COL_MODEL] >= 5.0)].copy()
    if pos_filter != "ALL": active = active[active['Pos'] == pos_filter].copy()
    if search_txt and str(search_txt).strip(): active = active[active['Player'].str.contains(str(search_txt).strip(), case=False, na=False) | active['Team'].str.contains(str(search_txt).strip(), case=False, na=False)].copy()
    cc = {"Receiving Yards": ('rec_yd', ['WR', 'TE', 'RB'], 18.5, 8.5, "Yds"), "Rushing Yards": ('rush_yd', ['RB', 'QB'], 16.5, 8.5, "Yds"), "Passing Yards": ('pass_yd', ['QB'], 42.0, 120.0, "Yds"), "Receptions (PPR)": ('receptions', ['WR', 'TE', 'RB'], 1.65, 1.5, "Rec")}
    cats = list(cc.keys()) if prop_category == "ALL PROP MARKETS" else [prop_category]
    rows = []
    for cl in cats:
        pk, vpos, bsig, mline, ustr = cc[cl]
        for _, r in active[active['Pos'].isin(vpos)].iterrows():
            mstat = float(r[f'Proj_{pk}']); lval = float(r[f'Mkt_{pk}'])
            if lval < mline and mstat < mline: continue
            sig = max(bsig * 0.65, bsig * (0.55 + 0.45 * (max(lval, mstat) / (mline * 2.5))))
            pover = (1.0 - normal_cdf((lval - mstat) / sig)) * 100.0
            side = "OVER" if mstat >= lval else "UNDER"
            wprob = round(float(np.clip(pover if side == "OVER" else (100.0 - pover), 50.1, 79.5)), 1)
            edge = round(wprob - 52.38, 1)
            if edge < float(min_edge_filter): continue
            rows.append({"Player": r['Player'], "Pos": r['Pos'], "Team": r['Team'], "Matchup": r['Matchup'], "Prop Market": cl, "Model Stat Proj": f"{mstat:.1f} {ustr}", "Market Line": f"{lval:.1f} {ustr}", "Recommended Bet": f"{side} {lval:.1f}", "Hit Prob %": f"{wprob:.1f}%", "Expected Value (+EV%)": f"{edge:+.1f}%", "Fair Odds": prob_to_american_odds(wprob), "Anytime TD %": f"{r['Anytime TD %']:.1f}%" if r['Pos'] != 'QB' else "N/A", "_ev": edge})
    cols = ["Player", "Pos", "Team", "Matchup", "Prop Market", "Model Stat Proj", "Market Line", "Recommended Bet", "Hit Prob %", "Expected Value (+EV%)", "Fair Odds", "Anytime TD %"]
    if not rows: return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)
    fdf = pd.DataFrame(rows).sort_values("_ev", ascending=False).drop(columns=["_ev"]).reset_index(drop=True)
    return fdf.head(10), fdf

def evaluate_custom_prop_bet(week_num, player_name, prop_market, custom_line, custom_odds):
    if not player_name: return "Select a player."
    wk_df = build_week_projections(int(week_num))
    p_rows = wk_df[wk_df['Player'] == player_name]
    if p_rows.empty: return "Player not found."
    r = p_rows.iloc[0]
    cmap = {"Receiving Yards": ('rec_yd', 18.5, "Yards", 140.0), "Rushing Yards": ('rush_yd', 16.5, "Yards", 140.0), "Passing Yards": ('pass_yd', 42.0, "Yards", 360.0), "Receptions (PPR)": ('receptions', 1.65, "Receptions", 12.0)}
    pk, sig, ustr, mscale = cmap.get(prop_market, ('rec_yd', 18.5, "Yards", 140.0))
    mproj = float(r[f'Proj_{pk}'])
    lv = float(custom_line) if custom_line else float(r[f'Mkt_{pk}'])
    pover = (1.0 - normal_cdf((lv - mproj) / sig)) * 100.0
    side = "OVER" if mproj >= lv else "UNDER"
    hprob = round(float(np.clip(pover if side == "OVER" else (100.0 - pover), 5.0, 92.0)), 1)
    iprob = round(american_odds_to_implied_prob(custom_odds) * 100.0, 1)
    edge = round(hprob - iprob, 1)
    bar = render_payoff_distribution_bar_html(f"{player_name} — {prop_market}", f"Line: {lv:.1f}", max(0.0, round(mproj - 1.28 * sig, 1)), mproj, lv, round(mproj + 1.28 * sig, 1), max_scale=max(mscale, mproj + 25.0), compact=True)
    return f'<div style="padding:12px; border:1px solid rgba(148,163,184,0.35); border-radius:10px; margin-top:8px;"><strong>Verdict: {side} {lv:.1f} ({edge:+.1f}% EV | True Hit Prob: {hprob:.1f}% vs {iprob:.1f}% Implied | Fair Odds: {prob_to_american_odds(hprob)})</strong>{bar}</div>'

def analyze_sleeper_league(league_id, week_num):
    wk_int = int(week_num)
    pr_cols = ["Rank", "Manager", "Record", COL_MODEL, COL_SLEEPER, COL_DIFF, "Live Banked Pts", "Playoff %", "Title %"]
    match_cols = ["Matchup", "Pre-Game Model Proj", "Live Score", "Win Odds", "#1 Matchup Decider"]
    if not league_id or not str(league_id).strip(): return "**Paste any public Sleeper League ID above**", pd.DataFrame(columns=pr_cols), "", pd.DataFrame(columns=match_cols)
    league_id = str(league_id).strip(); headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        ur = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/users", headers=headers, timeout=8)
        rr = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/rosters", headers=headers, timeout=8)
        mr = requests.get(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{wk_int}", headers=headers, timeout=8)
        if ur.status_code != 200 or rr.status_code != 200: return f"Could not fetch league {league_id}.", pd.DataFrame(columns=pr_cols), "", pd.DataFrame(columns=match_cols)
    except Exception as e: return f"Network notice: {e}", pd.DataFrame(columns=pr_cols), "", pd.DataFrame(columns=match_cols)

    users = {u['user_id']: u.get('display_name') or u.get('username') or "Manager" for u in ur.json()}
    rosters = rr.json(); m_map = {}
    if mr.status_code == 200 and isinstance(mr.json(), list):
        for m in mr.json():
            if m.get('roster_id') is not None: m_map[m['roster_id']] = m

    wk_df = build_week_projections(wk_int)
    by_gsis = {str(r['player_id']).strip(): r for _, r in wk_df.iterrows()}
    by_norm = {r['norm_name']: r for _, r in wk_df.iterrows()}

    t_rows = []
    for r in rosters:
        rid = r.get('roster_id'); mgr = users.get(r.get('owner_id'), f"Team {rid}")
        minfo = m_map.get(rid, {})
        starters = [str(x) for x in (minfo.get('starters') or r.get('starters') or []) if str(x) != "0"]
        live_pts = float(minfo.get('points') or 0.0)
        rset = r.get('settings', {}) or {}
        wins, losses = int(rset.get('wins', 0) or 0), int(rset.get('losses', 0) or 0)
        mtot, stot, vtot = 0.0, 0.0, 0.0; bdec, bsw = "Balanced Lineup", -1.0
        for pid in starters:
            if pid.isalpha(): mtot += 7.8; stot += 7.8; continue
            info = STATE['sleeper_pid_to_info'].get(pid)
            if not info: continue
            if info['pos'] in ['K', 'DEF']: mtot += 8.0; stot += 8.0; continue
            prow = by_gsis.get(info['gsis_id']) if info['gsis_id'] in by_gsis else by_norm.get(info['norm'])
            if prow is not None:
                mp = float(prow[COL_MODEL]); sp = float(prow[COL_SLEEPER])
                swing = round(float(prow['Boom (90%)']) - float(prow['Floor (10%)']), 1)
                mtot += mp; stot += sp; vtot += max(12.0, (swing / 2.563) ** 2)
                if swing > bsw: bsw = swing; bdec = f"{prow['Player']} ({prow['Floor (10%)']:.1f}–{prow['Boom (90%)']:.1f} PPR)"
        t_rows.append({'roster_id': rid, 'matchup_id': minfo.get('matchup_id'), 'Manager': mgr, 'Record': f"{wins}-{losses}", COL_MODEL: round(mtot, 2), COL_SLEEPER: round(stot, 2), COL_DIFF: round(mtot - stot, 2), 'Live Banked Pts': round(live_pts, 2), 'Playoff %': f"{min(99.0, max(5.0, round(50.0 + (wins - losses) * 12.5 + (mtot - 115.0) * 0.8, 1)))}%", 'Title %': f"{min(45.0, max(1.0, round(10.0 + (wins - losses) * 3.0 + (mtot - 115.0) * 0.35, 1)))}%", 'var': max(160.0, vtot), 'decider': bdec})
    pr_df = pd.DataFrame(t_rows).sort_values(by=COL_MODEL, ascending=False).reset_index(drop=True)
    pr_df.insert(0, "Rank", range(1, len(pr_df) + 1))

    mgroups = {}
    for t in t_rows:
        if t['matchup_id'] is not None: mgroups.setdefault(t['matchup_id'], []).append(t)
    tug = '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(360px, 1fr)); gap:12px; margin-bottom:14px;">'
    mrows = []
    for mid, pair in sorted(mgroups.items()):
        if len(pair) == 2:
            tA, tB = pair[0], pair[1]
            wpa = round(normal_cdf((tA[COL_MODEL] - tB[COL_MODEL]) / math.sqrt(tA['var'] + tB['var'])) * 100.0, 1)
            wpb = round(100.0 - wpa, 1)
            tug += f'<div style="padding:12px; border:1px solid rgba(148,163,184,0.35); border-radius:10px;"><div style="display:flex; justify-content:space-between; font-weight:700;"><span>{tA["Manager"]} ({wpa}%)</span><span>{tB["Manager"]} ({wpb}%)</span></div><div style="height:8px; background:#dc2626; border-radius:4px; overflow:hidden; margin:6px 0;"><div style="width:{wpa}%; background:#2563eb; height:100%;"></div></div><div style="font-size:0.78rem; opacity:0.8;">Deciders: {tA["decider"]} vs {tB["decider"]}</div></div>'
            mrows.append({"Matchup": f"{tA['Manager']} vs {tB['Manager']}", "Pre-Game Model Proj": f"{tA[COL_MODEL]:.2f} vs {tB[COL_MODEL]:.2f}", "Live Score": f"{tA['Live Banked Pts']:.1f} - {tB['Live Banked Pts']:.1f}", "Win Odds": f"{wpa}% / {wpb}%", "#1 Matchup Decider": f"{tA['decider']} | {tB['decider']}"})
    tug += '</div>'
    return f"**Sync Complete (Week {wk_int})**", pr_df[["Rank", "Manager", "Record", COL_MODEL, COL_SLEEPER, COL_DIFF, "Live Banked Pts", "Playoff %", "Title %"]], tug, pd.DataFrame(mrows) if mrows else pd.DataFrame(columns=match_cols)

def run_accuracy_benchmark(eval_week_choice):
    ensure_engine_initialized()
    lcw = STATE['latest_completed_week']
    weeks = list(range(2, lcw + 1)) if eval_week_choice == "All Completed 2026 Weeks (Weeks 2+)" else [int(eval_week_choice.replace("Week ", ""))]
    eval_df = pd.concat([build_week_projections(w).query("is_bye == False and is_out == False and `Actual PPR`.notna() and `Model's Projection` >= 5.0").assign(Week=w) for w in weeks], ignore_index=True)
    eval_df['Model Error'] = (eval_df[COL_MODEL] - eval_df['Actual PPR']).abs().round(2)
    eval_df['Sleeper Error'] = (eval_df[COL_SLEEPER] - eval_df['Actual PPR']).abs().round(2)
    non_ties = eval_df[eval_df['Model Error'] != eval_df['Sleeper Error']]
    wrate = (non_ties['Model Error'] < non_ties['Sleeper Error']).mean() * 100 if len(non_ties) > 0 else 50.0
    omae, smae = eval_df['Model Error'].mean(), eval_df['Sleeper Error'].mean()
    smd = f"### Head-to-Head Accuracy Scorecard ({eval_week_choice} — {len(eval_df)} Player-Weeks)\n| Metric | Model's Projection | Sleeper's Projection | Net Edge |\n| :--- | :---: | :---: | :---: |\n| **MAE** | **{omae:.2f} pts** | {smae:.2f} pts | **{smae - omae:+.2f} pts** |\n| **Win Rate** | **{wrate:.1f}%** | {100 - wrate:.1f}% | **{'Model Leads' if wrate >= 50 else 'Sleeper Leads'}** |"
    return smd, eval_df.sort_values(by='Actual PPR', ascending=False)[['Week', 'Player', 'Pos', 'Team', 'Matchup', COL_MODEL, COL_SLEEPER, COL_DIFF, 'Actual PPR', 'Model Error', 'Sleeper Error']].reset_index(drop=True)

def initialize_dashboard():
    ensure_engine_initialized()
    wk = STATE['default_target_week']
    banner, tbl = update_main_table(wk, True, True, True, "ALL", "ALL", "")
    booms, busts = get_top_outliers(wk)
    d_hi, d_lo = get_biggest_differers(wk, "ALL")
    top_p, all_p = build_player_props_board(wk, "ALL PROP MARKETS", "ALL", 0.0, "")
    p_names = STATE['active_player_names']
    c_html, c_sum, c_df = compare_multiple_players(wk, p_names[:3])
    eval_choices = ["All Completed 2026 Weeks (Weeks 2+)"] + [f"Week {w}" for w in range(2, STATE['latest_completed_week'] + 1)]
    acc_md, acc_df = run_accuracy_benchmark(eval_choices[0])
    cust_html = evaluate_custom_prop_bet(wk, p_names[0] if p_names else "", "Receiving Yards", 64.5, "-110")
    return (
        gr.update(value=str(wk)), banner, tbl, gr.update(value=str(wk)), booms, busts,
        gr.update(value=str(wk)), d_hi, d_lo, gr.update(value=str(wk)),
        gr.update(choices=p_names, value=p_names[:3]), c_html, c_sum, c_df,
        gr.update(value=str(wk)), gr.update(choices=p_names, value=p_names[0] if p_names else None), cust_html, top_p, all_p,
        gr.update(choices=eval_choices, value=eval_choices[0]), acc_md, acc_df
    )

with gr.Blocks(theme="soft", title="NFL Quantitative Scouting, League & Props Suite") as demo:
    gr.Markdown("# NFL Quantitative Scouting, League Intelligence & Player Props Suite (2026)")
    with gr.Tabs():
        with gr.TabItem("1. Player Projections & Scouting"):
            with gr.Tabs():
                with gr.TabItem("Weekly Projections & Visual Scouting Cards"):
                    with gr.Row():
                        wk_select = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value="11", label="NFL Week (1-18)", scale=1)
                        pos_sel = gr.Dropdown(choices=["ALL", "QB", "RB", "WR", "TE"], value="ALL", label="Position", scale=1)
                        outlook_sel = gr.Dropdown(choices=["ALL", "EXPECTED BASELINE", "HIGH-CEILING SKEW", "HIGH BUST RISK", "QUESTIONABLE"], value="ALL", label="Outlook Skew Filter", scale=1)
                        search_in = gr.Textbox(label="Search Player or Team", placeholder="e.g. Bijan, Henry...", scale=2)
                        refresh_p_btn = gr.Button("↻ Force Live Refresh", variant="secondary", scale=1)
                    with gr.Row():
                        hide_bye_chk = gr.Checkbox(value=True, label="Hide Bye Weeks")
                        hide_inj_chk = gr.Checkbox(value=True, label="Hide Injured / Out")
                        compact_chk = gr.Checkbox(value=True, label="Sleek Compact View")
                    status_banner = gr.Markdown("Loading live 2026 projections...")
                    with gr.Column(visible=False) as player_drawer:
                        close_drawer_btn = gr.Button("✖ Close Player Scouting Card", variant="secondary", size="sm")
                        drawer_html_view = gr.HTML()
                        drawer_logs = gr.DataFrame(label="2026 Weekly Game Logs", interactive=False, visible=False)
                    main_df_view = gr.DataFrame(label="Click Any Player Row Below to Open Their Visual Payoff Distribution Bar", interactive=False)
                    for ctrl in [wk_select, hide_bye_chk, hide_inj_chk, compact_chk, pos_sel, outlook_sel, search_in]:
                        ctrl.change(fn=update_main_table, inputs=[wk_select, hide_bye_chk, hide_inj_chk, compact_chk, pos_sel, outlook_sel, search_in], outputs=[status_banner, main_df_view])
                    refresh_p_btn.click(fn=force_refresh_main_table, inputs=[wk_select, hide_bye_chk, hide_inj_chk, compact_chk, pos_sel, outlook_sel, search_in], outputs=[status_banner, main_df_view])
                    main_df_view.select(fn=open_player_card_on_click, inputs=[main_df_view, wk_select], outputs=[player_drawer, drawer_html_view, drawer_logs])
                    close_drawer_btn.click(fn=close_player_scouting_drawer, inputs=None, outputs=[player_drawer, drawer_html_view, drawer_logs])
                with gr.TabItem("Top 5 Booms & Busts"):
                    with gr.Row():
                        bb_wk_select = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value="11", label="Target NFL Week")
                        bb_btn = gr.Button("Refresh Outliers", variant="primary")
                    boom_df = gr.DataFrame(label="Top 5 Starter Boom Candidates", interactive=False)
                    bust_df = gr.DataFrame(label="Top 5 Starter Bust Candidates", interactive=False)
                    bb_btn.click(fn=get_top_outliers, inputs=bb_wk_select, outputs=[boom_df, bust_df])
                    bb_wk_select.change(fn=get_top_outliers, inputs=bb_wk_select, outputs=[boom_df, bust_df])
                with gr.TabItem("Biggest Differers vs. Sleeper"):
                    with gr.Row():
                        diff_wk_select = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value="11", label="Target NFL Week")
                        diff_pos_select = gr.Dropdown(choices=["ALL", "QB", "RB", "WR", "TE"], value="ALL", label="Filter Position")
                        diff_btn = gr.Button("Refresh Discrepancies", variant="primary")
                    diff_higher_df = gr.DataFrame(label="Much Higher Than Sleeper", interactive=False)
                    diff_lower_df = gr.DataFrame(label="Much Lower Than Sleeper", interactive=False)
                    diff_btn.click(fn=get_biggest_differers, inputs=[diff_wk_select, diff_pos_select], outputs=[diff_higher_df, diff_lower_df])
                    diff_wk_select.change(fn=get_biggest_differers, inputs=[diff_wk_select, diff_pos_select], outputs=[diff_higher_df, diff_lower_df])
                    diff_pos_select.change(fn=get_biggest_differers, inputs=[diff_wk_select, diff_pos_select], outputs=[diff_higher_df, diff_lower_df])
        with gr.TabItem("2. League Command Center"):
            with gr.Row():
                league_id_in = gr.Textbox(value="", label="Sleeper League ID", placeholder="Paste public Sleeper League ID...")
                league_wk_in = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value="11", label="Target NFL Week")
                league_btn = gr.Button("Sync Real-Time League Data", variant="primary")
            league_summary = gr.Markdown("**Paste any public Sleeper League ID above**")
            league_tug_html = gr.HTML()
            league_pr_table = gr.DataFrame(label="Power Rankings", interactive=False)
            league_match_table = gr.DataFrame(label="Matchups", interactive=False)
            league_btn.click(fn=analyze_sleeper_league, inputs=[league_id_in, league_wk_in], outputs=[league_summary, league_pr_table, league_tug_html, league_match_table])
        with gr.TabItem("3. Multi-Player Comparison (2 to 6 Players)"):
            with gr.Row():
                comp_wk = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value="11", label="Target Week", scale=1)
                comp_players_sel = gr.Dropdown(choices=[], value=[], multiselect=True, max_choices=6, label="Select 2 to 6 Players", scale=3)
                comp_btn = gr.Button("Compare", variant="primary", scale=1)
            comp_cards_html = gr.HTML()
            comp_summary_md = gr.Markdown()
            comp_table_view = gr.DataFrame(label="Matrix", interactive=False)
            comp_btn.click(fn=compare_multiple_players, inputs=[comp_wk, comp_players_sel], outputs=[comp_cards_html, comp_summary_md, comp_table_view])
            comp_players_sel.change(fn=compare_multiple_players, inputs=[comp_wk, comp_players_sel], outputs=[comp_cards_html, comp_summary_md, comp_table_view])
        with gr.TabItem("4. Player Props & Sportsbook +EV Finder"):
            with gr.Row():
                prop_wk_sel = gr.Dropdown(choices=[str(w) for w in range(1, 19)], value="11", label="Target Week", scale=1)
                prop_cat_sel = gr.Dropdown(choices=["ALL PROP MARKETS", "Receiving Yards", "Rushing Yards", "Passing Yards", "Receptions (PPR)"], value="ALL PROP MARKETS", label="Market", scale=1)
                prop_pos_sel = gr.Dropdown(choices=["ALL", "QB", "RB", "WR", "TE"], value="ALL", label="Position", scale=1)
                prop_edge_sel = gr.Slider(minimum=-10.0, maximum=12.0, value=0.0, step=0.5, label="Min +EV%", scale=1)
                prop_search_in = gr.Textbox(label="Search Player", placeholder="e.g. Lamb...", scale=1)
            with gr.Accordion("Custom Sportsbook Prop Calculator", open=True):
                with gr.Row():
                    calc_player_sel = gr.Dropdown(choices=[], value=None, label="Player", scale=2)
                    calc_market_sel = gr.Dropdown(choices=["Receiving Yards", "Rushing Yards", "Passing Yards", "Receptions (PPR)"], value="Receiving Yards", label="Market", scale=2)
                    calc_line_in = gr.Number(value=64.5, label="Line", scale=1)
                    calc_odds_in = gr.Textbox(value="-110", label="Odds", scale=1)
                    calc_btn = gr.Button("Calculate", variant="secondary", scale=1)
                calc_out_html = gr.HTML()
            top_ev_props_table = gr.DataFrame(label="Top 10 +EV Props", interactive=False)
            all_props_table = gr.DataFrame(label="All Props", interactive=False)
            for p_ctrl in [prop_wk_sel, prop_cat_sel, prop_pos_sel, prop_edge_sel, prop_search_in]:
                p_ctrl.change(fn=build_player_props_board, inputs=[prop_wk_sel, prop_cat_sel, prop_pos_sel, prop_edge_sel, prop_search_in], outputs=[top_ev_props_table, all_props_table])
            calc_btn.click(fn=evaluate_custom_prop_bet, inputs=[prop_wk_sel, calc_player_sel, calc_market_sel, calc_line_in, calc_odds_in], outputs=calc_out_html)
        with gr.TabItem("5. Model Accuracy & Methodology"):
            eval_selector = gr.Dropdown(choices=["All Completed 2026 Weeks (Weeks 2+)"], value="All Completed 2026 Weeks (Weeks 2+)", label="Select Week(s)")
            acc_summary_md = gr.Markdown()
            acc_table = gr.DataFrame(label="Accuracy Comparison", interactive=False)
            eval_selector.change(fn=run_accuracy_benchmark, inputs=eval_selector, outputs=[acc_summary_md, acc_table])

    demo.load(fn=initialize_dashboard, inputs=None, outputs=[
        wk_select, status_banner, main_df_view, bb_wk_select, boom_df, bust_df,
        diff_wk_select, diff_higher_df, diff_lower_df, comp_wk, comp_players_sel,
        comp_cards_html, comp_summary_md, comp_table_view, prop_wk_sel, calc_player_sel,
        calc_out_html, top_ev_props_table, all_props_table, eval_selector, acc_summary_md, acc_table
    ])

port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
