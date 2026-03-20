#!/usr/bin/env python3
"""
Boston Red Sox 2025: Rotation → Closer Disorientation Analysis
==============================================================
For each starter in the rotation, measures how mechanically different
they are from the closer (Aroldis Chapman), then correlates that
disorientation score with Chapman's actual performance on those nights.

Outputs:
  boston_2025_dashboard.png   — macro view: release, movement,
                                 disorientation matrix, performance stats
  boston_2025_pitches.png     — pitch-level: similarity heatmap +
                                 side-by-side trajectory overlays
  boston_2025_sequencing.png  — approach & sequencing: first-pitch tendencies,
                                 count-based pitch selection, transition matrices

HOW TO RUN:
    python3 rotation_closer_analysis.py

HOW TO TUNE WEIGHTS:
    Edit the WEIGHTS dict below. Values are relative importance and are
    auto-normalized — set any to 0 to disable that dimension.

ADD MORE TEAMS:
    Drop additional {pitcher_id}_data.csv files in the same folder,
    update STARTERS / CLOSER, and re-run.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                   # saves PNG without needing a display
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import jensenshannon
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# ── optional pybaseball ───────────────────────────────────────────────────────
try:
    import pybaseball                   # pip install pybaseball  (not required)
    HAS_PYBASEBALL = True
except ImportError:
    HAS_PYBASEBALL = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  ── edit this section, then re-run
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
TEAM_LABEL = 'Boston Red Sox'
SEASON     = 2025

# ── Closer ───────────────────────────────────────────────────────────────────
CLOSER = {
    'id':          '547973',
    'name':        'Aroldis Chapman',
    'short':       'Chapman',
    'team_filter': 'BOS',       # only count games where he pitched for BOS
    'color':       '#FF4D4D',
}

# ── Rotation ─────────────────────────────────────────────────────────────────
# team_filter: restrict to games where this pitcher threw for that team.
# Set None to use all games (e.g. when a pitcher was only ever on one team).
STARTERS = [
    {'id': '676979', 'name': 'Garrett Crochet', 'short': 'Crochet',
     'team_filter': None,  'color': '#4FC3F7'},
    {'id': '678394', 'name': 'Brayan Bello',    'short': 'Bello',
     'team_filter': None,  'color': '#81C784'},
    {'id': '608337', 'name': 'Lucas Giolito',   'short': 'Giolito',
     'team_filter': None,  'color': '#FFB74D'},
    {'id': '621111', 'name': 'Walker Buehler',  'short': 'Buehler',
     'team_filter': 'BOS', 'color': '#CE93D8'},   # played for PHI earlier
]

# ── Disorientation Weights ────────────────────────────────────────────────────
# Relative importance — auto-normalized to sum to 1.
# Set 0 to disable any dimension entirely.
#
# MECHANICS  — what the batter sees from the mound
# APPROACH   — how the pitcher uses his stuff situationally
WEIGHTS = {
    # ── Mechanics ──────────────────────────────────────────────────────────────
    'Handedness Flip':      1.5,   # LHP vs RHP — biggest single visual shift
    'Arm Angle':            1.5,   # visual plane the ball comes from
    'Release Point':        1.2,   # where in the window the ball appears
    'Effective Velocity':   1.3,   # batter's timing is calibrated to this
    'Extension':            0.7,   # how far toward plate at release
    'Movement Profile':     1.2,   # how differently does the ball move overall
    'Spin Axis':            0.8,   # perceived rotation / trajectory shape
    # ── Approach (NEW) ─────────────────────────────────────────────────────────
    'Arsenal Overlap':      1.0,   # which pitch types exist in each arsenal
    'First Pitch Pattern':  1.4,   # how they open each at-bat (first pitch)
    'Two-Strike Approach':  1.3,   # pitch selection when ahead 0-2 / 1-2 / 2-2
    'Pitch Sequencing':     1.0,   # which pitch follows which (transition matrix)
}


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BG    = '#0D1117'
PANEL = '#161B22'
FG    = '#E6EDF3'
MUTED = '#8B949E'
GRIDC = '#30363D'

PITCH_COLORS = {
    '4-Seam Fastball': '#FF4D4D',
    'Sinker':          '#FFB347',
    'Cutter':          '#4FC3F7',
    'Sweeper':         '#81C1E8',
    'Slider':          '#CE93D8',
    'Changeup':        '#A5D6A7',
    'Split-Finger':    '#FFCC80',
    'Curveball':       '#B39DDB',
    'Knuckle Curve':   '#80CBC4',
}

# Pitch family groupings — similarity comparisons are constrained within families.
# A fastball can never be "most similar" to a breaking ball regardless of metrics.
PITCH_FAMILY = {
    '4-Seam Fastball': 'fastball',
    'Sinker':          'fastball',
    'Cutter':          'fastball',
    'Slider':          'breaking',
    'Sweeper':         'breaking',
    'Curveball':       'breaking',
    'Knuckle Curve':   'breaking',
    'Changeup':        'offspeed',
    'Split-Finger':    'offspeed',
    'Splitter':        'offspeed',
}

ABBREV = {
    '4-Seam Fastball': '4S',  'Sinker': 'SI',   'Cutter': 'CT',
    'Sweeper': 'SW',           'Slider': 'SL',   'Changeup': 'CH',
    'Split-Finger': 'SPL',     'Curveball': 'CB','Knuckle Curve': 'KC',
}

SWING_DESC = {
    'swinging_strike', 'swinging_strike_blocked', 'foul',
    'foul_bunt', 'foul_tip', 'hit_into_play', 'missed_bunt',
}
WHIFF_DESC = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
OUT_ZONES  = {11, 12, 13, 14}

NUM_COLS = [
    'release_speed', 'effective_speed', 'release_pos_x', 'release_pos_z',
    'release_pos_y', 'release_extension', 'release_spin_rate', 'spin_axis',
    'arm_angle', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'api_break_z_with_gravity', 'api_break_x_batter_in',
    'launch_speed', 'woba_value', 'woba_denom',
    'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
    'bat_score', 'fld_score', 'post_bat_score', 'zone',
    'balls', 'strikes', 'pitch_number',              # needed for approach dims
]


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _pitcher_team_col(df):
    """Infer which team the pitcher represented for each row."""
    return np.where(df['inning_topbot'] == 'Top', df['home_team'], df['away_team'])


def load_pitcher(pitcher_id, team_filter=None):
    path = os.path.join(DATA_DIR, f'{pitcher_id}_data.csv')
    if not os.path.exists(path):
        sys.exit(
            f'\n  ERROR: {path} not found.\n'
            f'  Download from baseballsavant.mlb.com/statcast_search\n'
            f'  and save as {pitcher_id}_data.csv in {DATA_DIR}\n'
        )
    df = pd.read_csv(path, low_memory=False)
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[df['pitch_name'].notna() & (df['pitch_name'].str.strip() != '')].copy()
    df['pitch_name'] = df['pitch_name'].str.strip()

    if team_filter:
        df['_pt'] = _pitcher_team_col(df)
        df = df[df['_pt'] == team_filter].copy()
        df.drop(columns=['_pt'], inplace=True)

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PITCHER FINGERPRINTING
# ═══════════════════════════════════════════════════════════════════════════════

def build_fingerprint(df):
    total = len(df)
    fp = {
        'name':           df['player_name'].iloc[0],
        'hand':           df['p_throws'].mode()[0],
        'total_pitches':  total,
        'arm_angle':      df['arm_angle'].median(),
        'release_z':      df['release_pos_z'].median(),
        'release_x':      df['release_pos_x'].median(),
        'extension':      df['release_extension'].median(),
        'velo':           df['release_speed'].median(),
        'effective_velo': df['effective_speed'].median(),
        'spin_rate':      df['release_spin_rate'].median(),
        'spin_axis':      df['spin_axis'].median(),
        'pfx_x':          df['pfx_x'].median(),
        'pfx_z':          df['pfx_z'].median(),
        'pitch_mix':     (df['pitch_name'].value_counts(normalize=True) * 100).round(2),
    }

    # ── Per-pitch-type stats ──────────────────────────────────────────────────
    by_pitch = {}
    for pitch, g in df.groupby('pitch_name'):
        if len(g) < 15:
            continue
        kin = {c: g[c].mean() if c in g.columns else np.nan
               for c in ('vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
                         'release_pos_x', 'release_pos_z', 'release_pos_y')}
        by_pitch[pitch] = {
            'count':     len(g),
            'pct':       100 * len(g) / total,
            'velo':      g['release_speed'].mean(),
            'spin':      g['release_spin_rate'].mean(),
            'spin_axis': g['spin_axis'].mean(),
            'pfx_x_in':  g['pfx_x'].mean() * 12,
            'pfx_z_in':  g['pfx_z'].mean() * 12,
            'plate_x':   g['plate_x'].mean(),
            'plate_z':   g['plate_z'].mean(),
            **kin,
        }
    fp['by_pitch'] = by_pitch

    # ── Approach dimensions ───────────────────────────────────────────────────
    def _count_mix(balls_list, strikes_list, min_n=25):
        """Pitch-type distribution for pitches thrown in a specific count group."""
        mask = df['balls'].isin(balls_list) & df['strikes'].isin(strikes_list)
        sub  = df[mask]
        if len(sub) < min_n:
            return pd.Series(dtype=float)
        return (sub['pitch_name'].value_counts(normalize=True) * 100).round(2)

    # First pitch of each at-bat
    first = df[df['pitch_number'] == 1]
    fp['first_pitch_mix'] = (
        (first['pitch_name'].value_counts(normalize=True) * 100).round(2)
        if len(first) >= 15 else pd.Series(dtype=float)
    )

    # Pitcher's count (0-2, 1-2, 2-2) — "finishing" pitches
    fp['two_strike_mix'] = _count_mix([0, 1, 2, 3], [2])

    # Hitter's count (2-0, 3-0, 3-1) — must-throw-strike situations
    fp['hitters_count_mix'] = _count_mix([2, 3], [0, 1])

    # Breakdown by count bucket (for visualization)
    fp['count_mixes'] = {
        'First Pitch':    fp['first_pitch_mix'],
        '0-2 / 1-2 / 2-2 (Ahead)':  fp['two_strike_mix'],
        '2-0 / 3-0 / 3-1 (Behind)': fp['hitters_count_mix'],
        'Overall':        fp['pitch_mix'],
    }

    # Pitch transition matrix — what follows what within the same at-bat
    df_s = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).copy()
    df_s['_prev'] = df_s.groupby(
        ['game_pk', 'at_bat_number'])['pitch_name'].shift(1)
    trans = df_s.dropna(subset=['_prev'])
    fp['transition_matrix'] = (
        pd.crosstab(trans['_prev'], trans['pitch_name'], normalize='index')
        if len(trans) >= 30 else pd.DataFrame()
    )

    return fp


# ═══════════════════════════════════════════════════════════════════════════════
#  DISORIENTATION SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_weights(w):
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def _n(v, lo, hi):
    """Clip-normalize v to [0, 1] given expected range [lo, hi]."""
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _mix_js(m1, m2):
    """Jensen-Shannon divergence between two pitch-mix Series. Returns 0.5 if
    either is empty (unknown / uncertain, not identical nor opposite)."""
    if m1 is None or m2 is None or len(m1) == 0 or len(m2) == 0:
        return 0.5
    all_p = sorted(set(m1.index) | set(m2.index))
    v1 = np.array([m1.get(p, 0.0) for p in all_p], float)
    v2 = np.array([m2.get(p, 0.0) for p in all_p], float)
    if v1.sum() < 1e-9 or v2.sum() < 1e-9:
        return 0.5
    return float(jensenshannon(v1 / v1.sum(), v2 / v2.sum()))


def _transition_dist(tm1, tm2):
    """Average per-row JS divergence between two pitch transition matrices.
    Each row = 'given this pitch was just thrown, what comes next?'"""
    if tm1.empty or tm2.empty:
        return 0.5
    all_from = sorted(set(tm1.index) | set(tm2.index))
    all_to   = sorted(set(tm1.columns) | set(tm2.columns))
    jsds = []
    for prev in all_from:
        r1 = np.array([tm1.loc[prev, nx] if prev in tm1.index and nx in tm1.columns
                        else 0.0 for nx in all_to], float)
        r2 = np.array([tm2.loc[prev, nx] if prev in tm2.index and nx in tm2.columns
                        else 0.0 for nx in all_to], float)
        s1, s2 = r1.sum(), r2.sum()
        if s1 < 1e-9 and s2 < 1e-9:
            continue
        # If one pitcher never throws this pitch, treat their "next pitch"
        # distribution as uniform (maximum uncertainty) rather than skip entirely
        if s1 < 1e-9: r1 = np.ones(len(all_to)) / len(all_to)
        if s2 < 1e-9: r2 = np.ones(len(all_to)) / len(all_to)
        jsds.append(float(jensenshannon(r1 / r1.sum(), r2 / r2.sum())))
    return float(np.mean(jsds)) if jsds else 0.5


def compute_disorientation(fp_s, fp_c, nw):
    d = {}

    # ── Mechanics ──────────────────────────────────────────────────────────────
    d['Handedness Flip']    = 0.0 if fp_s['hand'] == fp_c['hand'] else 1.0
    d['Arm Angle']          = _n(abs(fp_s['arm_angle']      - fp_c['arm_angle']),      0, 40)
    rel_dist = np.sqrt((fp_s['release_x'] - fp_c['release_x'])**2 +
                       (fp_s['release_z'] - fp_c['release_z'])**2)
    d['Release Point']      = _n(rel_dist, 0, 2.5)
    d['Effective Velocity'] = _n(abs(fp_s['effective_velo'] - fp_c['effective_velo']), 0, 12)
    d['Extension']          = _n(abs(fp_s['extension']      - fp_c['extension']),      0, 2.0)

    def _centroid(fp):
        pts = fp['by_pitch']
        w   = sum(v['pct'] for v in pts.values())
        if w == 0:
            return np.zeros(2)
        return np.array([
            sum(v['pct'] * v['pfx_x_in'] for v in pts.values()) / w,
            sum(v['pct'] * v['pfx_z_in'] for v in pts.values()) / w,
        ])
    d['Movement Profile']   = _n(float(np.linalg.norm(_centroid(fp_s) - _centroid(fp_c))), 0, 20)

    ax_delta = abs(fp_s['spin_axis'] - fp_c['spin_axis'])
    d['Spin Axis']          = _n(min(ax_delta, 360 - ax_delta), 0, 90)

    # ── Approach (sequencing & count-based) ────────────────────────────────────
    # Overall arsenal (which pitch types and in what proportions)
    d['Arsenal Overlap']    = _mix_js(fp_s['pitch_mix'], fp_c['pitch_mix'])

    # How each pitcher opens at-bats (first pitch — batter's initial expectation)
    d['First Pitch Pattern'] = _mix_js(
        fp_s.get('first_pitch_mix'), fp_c.get('first_pitch_mix'))

    # What each pitcher throws when ahead 0-2 / 1-2 / 2-2
    # (batter calibrates to starter's finishing pitches; closer may be totally different)
    d['Two-Strike Approach'] = _mix_js(
        fp_s.get('two_strike_mix'), fp_c.get('two_strike_mix'))

    # Pitch-sequencing patterns: do they tunnel the same pairs? Follow fastballs
    # with the same off-speed? Average JS divergence across all "prev-pitch" rows.
    d['Pitch Sequencing']   = _transition_dist(
        fp_s.get('transition_matrix', pd.DataFrame()),
        fp_c.get('transition_matrix', pd.DataFrame()),
    )

    return {
        'dimensions': d,
        'composite':  sum(d[k] * nw[k] for k in nw),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  GAME LINKING
# ═══════════════════════════════════════════════════════════════════════════════

def link_games(df_closer, starter_dfs):
    """
    Match each closer game_pk to whichever starter in our set pitched
    that game. Returns {game_pk: starter_id}.
    Ties broken by pitch count (actual starter > opener/reliever).
    """
    closer_pks  = set(df_closer['game_pk'].unique())
    game_to = {}                                    # {game_pk: (starter_id, n_pitches)}

    for sid, df in starter_dfs.items():
        for gpk, grp in df.groupby('game_pk'):
            if gpk not in closer_pks:
                continue
            n = len(grp)
            if gpk not in game_to or n > game_to[gpk][1]:
                game_to[gpk] = (sid, n)

    return {pk: sid for pk, (sid, _) in game_to.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-GAME CLOSER METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_game_metrics(df):
    n       = len(df)
    bf      = df['at_bat_number'].nunique()
    k       = df['events'].eq('strikeout').sum()
    bb      = df['events'].eq('walk').sum()
    hits    = df['events'].isin({'single', 'double', 'triple', 'home_run'}).sum()
    hbp     = df['events'].eq('hit_by_pitch').sum()

    swings  = df['description'].isin(SWING_DESC).sum()
    whiffs  = df['description'].isin(WHIFF_DESC).sum()
    called  = df['description'].eq('called_strike').sum()

    # Chase rate: swings on out-of-zone pitches / total out-of-zone pitches
    ooz     = df['zone'].isin(OUT_ZONES)
    ooz_n   = int(ooz.sum())
    ooz_sw  = int(df.loc[ooz, 'description'].isin(SWING_DESC).sum()) if ooz_n else 0
    chase   = ooz_sw / ooz_n if ooz_n else np.nan

    # Runs allowed: score change during appearance
    runs = np.nan
    if 'bat_score' in df.columns and 'post_bat_score' in df.columns:
        first = df['bat_score'].dropna()
        last  = df['post_bat_score'].dropna()
        if len(first) and len(last):
            runs = float(max(0.0, last.iloc[-1] - first.iloc[0]))

    # Exit velocity / hard-hit
    ip       = df['launch_speed'].notna()
    n_ip     = int(ip.sum())
    avg_ev   = float(df.loc[ip, 'launch_speed'].mean())  if n_ip else np.nan
    hard_pct = float((df.loc[ip, 'launch_speed'] >= 95).sum() / n_ip) if n_ip else np.nan

    # wOBA against
    dm   = df['woba_denom'].gt(0)
    woba = float(df.loc[dm, 'woba_value'].sum() / df.loc[dm, 'woba_denom'].sum()) \
           if dm.any() else np.nan

    return {
        'pitches':       n,
        'batters_faced': bf,
        'k_pct':         k    / bf     if bf     else np.nan,
        'bb_pct':        bb   / bf     if bf     else np.nan,
        'hits_per_bf':   hits / bf     if bf     else np.nan,
        'runs_allowed':  runs,
        'whiff_pct':     whiffs / swings if swings else np.nan,
        'csw_pct':       (called + whiffs) / n if n else np.nan,
        'chase_rate':    chase,
        'woba_against':  woba,
        'avg_ev':        avg_ev,
        'hard_hit_pct':  hard_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PITCH SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════

def pitch_similarity_matrix(fp_s, fp_c):
    """
    Return (long_df, pivot) of similarity scores (0 = opposite, 1 = identical).
    Metrics: velocity, pfx movement, spin axis, zone location.

    Cross-family pairs (fastball vs breaking, etc.) are hard-capped at a
    similarity of 0.30 — a slider and a 4-seam can never be 'most similar'
    regardless of how their numbers accidentally line up.
    """
    rows = []
    for sp, ss in fp_s['by_pitch'].items():
        for cp, cs in fp_c['by_pitch'].items():
            velo_d = _n(abs(ss['velo']      - cs['velo']),      0, 15)
            pfx_d  = _n(np.linalg.norm([ss['pfx_x_in'] - cs['pfx_x_in'],
                                         ss['pfx_z_in'] - cs['pfx_z_in']]), 0, 25)
            axis_d = abs(ss['spin_axis'] - cs['spin_axis'])
            axis_d = _n(min(axis_d, 360 - axis_d), 0, 90)
            loc_d  = _n(np.linalg.norm([ss['plate_x'] - cs['plate_x'],
                                         ss['plate_z'] - cs['plate_z']]), 0, 3.0)
            dissim = float(np.nanmean([velo_d, pfx_d, axis_d, loc_d]))

            fam_s = PITCH_FAMILY.get(sp, 'unknown')
            fam_c = PITCH_FAMILY.get(cp, 'unknown')
            same_fam = (fam_s == fam_c and fam_s != 'unknown')

            # Hard cap: cross-family pairs can never score above 0.30 similarity.
            # This prevents a sinker from ever being called "most similar" to a
            # slider just because their average location happened to overlap.
            if not same_fam:
                dissim = max(dissim, 0.70)

            rows.append({
                'starter_pitch': sp,
                'closer_pitch':  cp,
                'dissimilarity': dissim,
                'similarity':    1.0 - dissim,
                'same_family':   same_fam,
                'family_s':      fam_s,
                'family_c':      fam_c,
            })

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index='starter_pitch', columns='closer_pitch',
        values='similarity', aggfunc='first',
    )
    return df, pivot


# ═══════════════════════════════════════════════════════════════════════════════
#  PITCH TRAJECTORY  (batter's view, from kinematic parameters)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_trajectory(pitch_stats, n_pts=80):
    """
    Reconstruct x (horizontal) and z (vertical) positions of the ball at every
    point from release to home plate, viewed from the batter's perspective.

    Uses Statcast kinematic columns: vx0, vy0, vz0, ax, ay, az,
    release_pos_x, release_pos_z, release_pos_y.

    Returns (x_arr, z_arr, y_arr) where y is distance from plate in feet,
    or (None, None, None) if data is unavailable.
    """
    needed = ('vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
              'release_pos_x', 'release_pos_z', 'release_pos_y')
    vals = {k: pitch_stats.get(k, np.nan) for k in needed}
    if any(np.isnan(v) for v in vals.values()):
        return None, None, None

    x0, z0, y0 = vals['release_pos_x'], vals['release_pos_z'], vals['release_pos_y']
    vx, vy, vz = vals['vx0'],           vals['vy0'],           vals['vz0']
    ax, ay, az = vals['ax'],            vals['ay'],            vals['az']

    # Solve y(t) = 0 for t_plate via quadratic: 0.5*ay*t² + vy*t + y0 = 0
    disc = vy**2 - 2.0 * ay * y0
    if disc < 0:
        return None, None, None
    t_plate = (-vy - np.sqrt(disc)) / ay
    if t_plate <= 0:
        return None, None, None

    t = np.linspace(0, t_plate, n_pts)
    x_arr = x0 + vx * t + 0.5 * ax * t**2
    z_arr = z0 + vz * t + 0.5 * az * t**2
    y_arr = y0 + vy * t + 0.5 * ay * t**2   # distance from plate (decreasing)
    return x_arr, z_arr, y_arr


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(results_df, dis, fp_starters, fp_closer, starters):
    cname = fp_closer['name']
    print(f'\n{"═"*76}')
    print(f'  {TEAM_LABEL} {SEASON}  ·  Chapman Performance by Starter')
    print(f'  Closer: {cname}  —  {fp_closer["hand"]}HP  —  {fp_closer["total_pitches"]} pitches')
    print(f'{"═"*76}')
    hdr = (f'{"Starter":<12} {"Hand":<5} {"Dis.":<6} {"n":<4} '
           f'{"K%":<6} {"Whiff":<7} {"Chase":<7} '
           f'{"H/BF":<6} {"Runs":<6} {"wOBA":<6} {"AvgEV":<6}')
    print(hdr)
    print('─' * 76)
    for s in starters:
        sub = results_df[results_df['starter_id'] == s['id']]
        fp  = fp_starters[s['id']]
        dis_c = dis[s['id']]['composite']
        if len(sub) == 0:
            print(f'{s["short"]:<12} {fp["hand"]}HP   {dis_c:<6.3f} 0    (no matched games)')
            continue
        print(
            f'{s["short"]:<12} {fp["hand"]}HP   {dis_c:<6.3f} {len(sub):<4} '
            f'{sub["k_pct"].mean():<6.3f} {sub["whiff_pct"].mean():<7.3f} '
            f'{sub["chase_rate"].mean():<7.3f} {sub["hits_per_bf"].mean():<6.3f} '
            f'{sub["runs_allowed"].mean():<6.1f} {sub["woba_against"].mean():<6.3f} '
            f'{sub["avg_ev"].mean():<6.1f}'
        )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sa(ax, grid=True):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRIDC)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid:
        ax.grid(color=GRIDC, linewidth=0.5, alpha=0.8, zorder=0)


def _ab(p):
    return ABBREV.get(p, p[:3])


def _pc(p):
    return PITCH_COLORS.get(p, '#9E9E9E')


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — MACRO DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def build_fig1(results_df, dis, fp_starters, fp_closer,
               starters, df_starters, df_closer):
    cname = fp_closer['name'].split(',')[0]
    fig   = plt.figure(figsize=(22, 14), facecolor=BG)

    fig.text(0.5, 0.978,
             f'{TEAM_LABEL} {SEASON}  ·  Rotation → Closer Disorientation',
             ha='center', va='top', fontsize=20, fontweight='bold',
             color=FG, fontfamily='monospace')
    fig.text(0.5, 0.952,
             f'How different was {cname} from each starter — '
             f'and did the disorientation help his performance?',
             ha='center', va='top', fontsize=10.5, color=MUTED)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.05, right=0.97, top=0.93, bottom=0.06,
        hspace=0.38, wspace=0.30,
    )

    _f1_release   (fig.add_subplot(gs[0, 0]), starters, df_starters, fp_starters,
                   df_closer, fp_closer, cname)
    _f1_movement  (fig.add_subplot(gs[0, 1]), starters, fp_starters, fp_closer, cname)
    _f1_heatmap   (fig.add_subplot(gs[0, 2]), starters, dis)
    _f1_perf_bars (fig.add_subplot(gs[1, 0]), starters, results_df, cname)
    _f1_scatter   (fig.add_subplot(gs[1, 1]), starters, results_df,
                   'k_pct', 'K%', f'{cname}  K%  vs  Disorientation Score')
    _f1_scatter   (fig.add_subplot(gs[1, 2]), starters, results_df,
                   'woba_against', 'wOBA Against',
                   f'{cname}  wOBA Against  vs  Disorientation Score')

    out = os.path.join(DATA_DIR, f'boston_{SEASON}_dashboard.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Figure 1 saved → {out}')


def _f1_release(ax, starters, df_starters, fp_starters,
                df_closer, fp_closer, cname):
    _sa(ax)
    for s in starters:
        df = df_starters[s['id']]
        ax.scatter(df['release_pos_x'], df['release_pos_z'],
                   s=4, color=s['color'], alpha=0.07, zorder=2)
        fp = fp_starters[s['id']]
        ax.scatter(fp['release_x'], fp['release_z'],
                   s=110, color=s['color'], zorder=5, marker='o',
                   edgecolors='white', linewidths=1.2, label=s['short'])

    ax.scatter(df_closer['release_pos_x'], df_closer['release_pos_z'],
               s=4, color=CLOSER['color'], alpha=0.07, zorder=2)
    ax.scatter(fp_closer['release_x'], fp_closer['release_z'],
               s=160, color=CLOSER['color'], zorder=6, marker='*',
               edgecolors='white', linewidths=1.2, label=cname)

    ax.set_xlabel("Horizontal position, ft  (catcher's view)", color=MUTED, fontsize=8)
    ax.set_ylabel('Release height, ft', color=MUTED, fontsize=8)
    ax.set_title('Release Window\n(dots = all pitches, marker = median)',
                 color=FG, fontsize=10, fontweight='bold', pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRIDC,
              labelcolor=FG, loc='lower right')


def _f1_movement(ax, starters, fp_starters, fp_closer, cname):
    _sa(ax)
    for s in starters:
        fp = fp_starters[s['id']]
        for pitch, ps in fp['by_pitch'].items():
            ax.scatter(ps['pfx_x_in'], ps['pfx_z_in'],
                       s=max(40, ps['pct'] * 8), color=_pc(pitch),
                       edgecolors=s['color'], linewidths=1.6,
                       marker='o', zorder=5, alpha=0.9)
            ax.annotate(f'{s["short"][0]}·{_ab(pitch)}',
                        (ps['pfx_x_in'], ps['pfx_z_in']),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=6, color=s['color'])
    for pitch, ps in fp_closer['by_pitch'].items():
        ax.scatter(ps['pfx_x_in'], ps['pfx_z_in'],
                   s=max(55, ps['pct'] * 8), color=_pc(pitch),
                   edgecolors=CLOSER['color'], linewidths=2.0,
                   marker='*', zorder=6, alpha=0.95)
        ax.annotate(f'C·{_ab(pitch)}',
                    (ps['pfx_x_in'], ps['pfx_z_in']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=6, color=CLOSER['color'])

    ax.axhline(0, color=GRIDC, lw=1)
    ax.axvline(0, color=GRIDC, lw=1)
    ax.set_xlabel("H. Break, in.  (pitcher's POV)", color=MUTED, fontsize=8)
    ax.set_ylabel('V. Break, in.  (vs gravity)',    color=MUTED, fontsize=8)
    ax.set_title('Pitch Movement Profile\n(dot size = usage %)',
                 color=FG, fontsize=10, fontweight='bold', pad=6)

    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=s['color'], markersize=8,
                      label=s['short']) for s in starters]
    handles.append(Line2D([0], [0], marker='*', color='w',
                           markerfacecolor=CLOSER['color'],
                           markersize=10, label=cname))
    ax.legend(handles=handles, fontsize=7.5, facecolor=PANEL,
              edgecolor=GRIDC, labelcolor=FG)


def _f1_heatmap(ax, starters, dis):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRIDC)
    ax.tick_params(colors=MUTED, labelsize=8)

    dims      = list(WEIGHTS.keys())
    row_lbls  = [s['short'] for s in starters]
    data      = np.array([[dis[s['id']]['dimensions'][d] for d in dims]
                           for s in starters])
    comps     = np.array([dis[s['id']]['composite'] for s in starters])
    full      = np.hstack([data, comps.reshape(-1, 1)])
    col_lbls  = dims + ['COMPOSITE']

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'dis', ['#4FC3F7', '#F9A825', '#E53935'])
    ax.imshow(full, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(col_lbls)))
    ax.set_xticklabels(
        [c.replace(' ', '\n') for c in col_lbls],
        rotation=0, ha='center', fontsize=6.2, color=FG,
    )
    ax.set_yticks(range(len(row_lbls)))
    ax.set_yticklabels(row_lbls, fontsize=9, color=FG)

    for ri, row in enumerate(full):
        for ci, v in enumerate(row):
            bold = (ci == len(col_lbls) - 1)
            ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                    fontsize=7.5,
                    fontweight='bold' if bold else 'normal',
                    color='#111' if v > 0.55 else FG)

    # Vertical separator before composite column
    ax.axvline(len(dims) - 0.5, color=FG, lw=1.5, alpha=0.5)
    ax.set_title(
        'Disorientation Matrix\n(0 = identical  →  1 = maximally different)',
        color=FG, fontsize=10, fontweight='bold', pad=6,
    )


def _f1_perf_bars(ax, starters, results_df, cname):
    _sa(ax)
    metrics = [
        ('k_pct',        'K%',      '#4FC3F7'),
        ('whiff_pct',    'Whiff%',  '#81C784'),
        ('chase_rate',   'Chase%',  '#FFB74D'),
        ('hits_per_bf',  'H/BF',    '#FF8A65'),
        ('woba_against', 'wOBA',    '#CE93D8'),
    ]
    x     = np.arange(len(starters))
    bw    = 0.15
    off_0 = -(len(metrics) - 1) / 2 * bw

    for mi, (col, lbl, color) in enumerate(metrics):
        vals = []
        for s in starters:
            sub = results_df[results_df['starter_id'] == s['id']][col].dropna()
            vals.append(float(sub.mean()) if len(sub) else np.nan)
        offset = off_0 + mi * bw
        ax.bar(x + offset, vals, bw, label=lbl,
               color=color, edgecolor=BG, linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([s['short'] for s in starters], color=FG, fontsize=9)
    ax.set_ylabel('Rate', color=MUTED, fontsize=8)
    ax.set_title(f'{cname}  —  Performance Breakdown by Starter',
                 color=FG, fontsize=10, fontweight='bold', pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRIDC,
              labelcolor=FG, ncol=3)


def _f1_scatter(ax, starters, results_df, metric, ylabel, title):
    _sa(ax)
    for s in starters:
        sub = results_df[results_df['starter_id'] == s['id']][
            ['disorientation', metric]
        ].dropna()
        if len(sub) == 0:
            continue
        ax.scatter(sub['disorientation'], sub[metric],
                   s=55, color=s['color'], alpha=0.82, zorder=4,
                   label=s['short'], edgecolors='white', linewidths=0.5)

    valid = results_df[['disorientation', metric]].dropna()
    if len(valid) >= 4:
        slope, intercept, r, p, _ = linregress(
            valid['disorientation'], valid[metric])
        xs = np.linspace(valid['disorientation'].min(),
                         valid['disorientation'].max(), 100)
        ax.plot(xs, slope * xs + intercept,
                color='white', lw=1.5, ls='--', alpha=0.55, zorder=3)
        ax.text(0.05, 0.93, f'r = {r:+.2f}   p = {p:.2f}',
                transform=ax.transAxes, color=MUTED, fontsize=8)
        dir_lbl = 'more similar → better' if slope < 0 else 'more different → better'
        ax.text(0.05, 0.86, dir_lbl,
                transform=ax.transAxes, color=MUTED, fontsize=7.5)

    ax.set_xlabel('Disorientation Score  (higher = more different)',
                  color=MUTED, fontsize=8)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=8)
    ax.set_title(title, color=FG, fontsize=10, fontweight='bold', pad=6)
    ax.legend(fontsize=7.5, facecolor=PANEL, edgecolor=GRIDC, labelcolor=FG)


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — PITCH ANALYSIS  (similarity + trajectories)
# ═══════════════════════════════════════════════════════════════════════════════

def build_fig2(starters, fp_starters, fp_closer):
    cname = fp_closer['name'].split(',')[0]
    n_s   = len(starters)

    # Each starter gets one row of 3 panels:
    #   [similarity heatmap | most-similar trajectory | most-dissimilar trajectory]
    fig = plt.figure(figsize=(22, 6 * n_s + 1), facecolor=BG)
    fig.text(0.5, 0.998,
             f'{TEAM_LABEL} {SEASON}  ·  Pitch Similarity & Trajectory Analysis',
             ha='center', va='top', fontsize=18, fontweight='bold',
             color=FG, fontfamily='monospace')
    fig.text(0.5, 0.984,
             f'Similarity scores and batter\'s-eye view of pitch paths '
             f'for each starter vs {cname}',
             ha='center', va='top', fontsize=10, color=MUTED)

    top_pad    = 0.97
    row_height = (top_pad - 0.03) / n_s
    outer = gridspec.GridSpec(
        n_s, 1, figure=fig,
        top=top_pad, bottom=0.03, left=0.04, right=0.97,
        hspace=0.50,
    )

    for ri, s in enumerate(starters):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[ri], wspace=0.28)
        fp_s   = fp_starters[s['id']]
        sim_df, sim_pivot = pitch_similarity_matrix(fp_s, fp_closer)

        _f2_sim_heatmap(fig.add_subplot(inner[0]), s, fp_s, fp_closer,
                        sim_df, sim_pivot, cname)
        _f2_trajectory (fig.add_subplot(inner[1]), s, fp_s, fp_closer,
                        sim_df, most_similar=True,  cname=cname)
        _f2_trajectory (fig.add_subplot(inner[2]), s, fp_s, fp_closer,
                        sim_df, most_similar=False, cname=cname)

    out = os.path.join(DATA_DIR, f'boston_{SEASON}_pitch_analysis.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Figure 2 saved → {out}')


def _f2_sim_heatmap(ax, s, fp_s, fp_c, sim_df, sim_pivot, cname):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRIDC)
    ax.tick_params(colors=MUTED, labelsize=8)

    if sim_pivot.empty:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                color=MUTED, transform=ax.transAxes)
        return

    row_pitches = list(sim_pivot.index)
    col_pitches = list(sim_pivot.columns)
    mat         = sim_pivot.values
    row_lbls    = [_ab(p) for p in row_pitches]
    col_lbls    = [_ab(p) for p in col_pitches]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'sim', ['#E53935', '#F9A825', '#43A047'])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(col_lbls)))
    ax.set_xticklabels(col_lbls, color=CLOSER['color'], fontsize=8.5)
    ax.set_yticks(range(len(row_lbls)))
    ax.set_yticklabels(row_lbls, color=s['color'], fontsize=8.5)

    # Build a lookup for same_family flag from sim_df
    fam_lookup = {}
    for _, row in sim_df.iterrows():
        fam_lookup[(row['starter_pitch'], row['closer_pitch'])] = row['same_family']

    for ri, sp in enumerate(row_pitches):
        for ci, cp in enumerate(col_pitches):
            v       = mat[ri, ci]
            same_f  = fam_lookup.get((sp, cp), True)
            # Cross-family cells: show score dimmed + diagonal slash marker
            txt_col = '#111' if (v > 0.5 and same_f) else (MUTED if not same_f else FG)
            ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                    fontsize=8.5, fontweight='bold' if same_f else 'normal',
                    color=txt_col, alpha=0.55 if not same_f else 1.0)
            if not same_f:
                # Dim overlay to signal "category mismatch — comparison invalid"
                ax.add_patch(mpatches.Rectangle(
                    (ci - 0.5, ri - 0.5), 1, 1,
                    facecolor='black', alpha=0.35, zorder=3))

    ax.set_xlabel(f'{cname} pitches  ▶', color=CLOSER['color'], fontsize=8)
    ax.set_ylabel(f'◀  {s["short"]} pitches', color=s['color'], fontsize=8)
    ax.set_title(
        f'{s["short"]}  vs  {cname}\n'
        f'Pitch Similarity  (dimmed = cross-family, excluded from "most similar")',
        color=FG, fontsize=9, fontweight='bold', pad=6,
    )


def _f2_trajectory(ax, s, fp_s, fp_c, sim_df, most_similar, cname):
    _sa(ax, grid=False)

    if sim_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                color=MUTED, transform=ax.transAxes)
        return

    tag = 'Most Similar' if most_similar else 'Most Dissimilar'

    if most_similar:
        # Only compare within the same pitch family (fastball↔fastball, etc.)
        pool = sim_df[sim_df['same_family'] == True]
        if pool.empty:
            pool = sim_df          # fallback if no same-family pairs exist
        row = pool.sort_values('similarity', ascending=False).iloc[0]
    else:
        # Most dissimilar can be any pair — cross-family pairs will naturally
        # dominate here since they're capped at 0.30 similarity
        row = sim_df.sort_values('similarity', ascending=True).iloc[0]

    sp_name  = row['starter_pitch']
    cp_name  = row['closer_pitch']
    sim_val  = row['similarity']

    sp_stats = fp_s['by_pitch'].get(sp_name)
    cp_stats = fp_c['by_pitch'].get(cp_name)

    if sp_stats is None or cp_stats is None:
        ax.text(0.5, 0.5, f'{tag}\n(data unavailable)',
                ha='center', va='center', color=MUTED, transform=ax.transAxes)
        return

    x_s, z_s, y_s = compute_trajectory(sp_stats)
    x_c, z_c, y_c = compute_trajectory(cp_stats)

    def _draw(x_arr, z_arr, y_arr, color, label, lw, ls):
        if x_arr is None:
            return
        n = len(x_arr)
        # Fade in as ball approaches plate (alpha 0.25 → 1.0)
        for i in range(n - 1):
            alpha = 0.25 + 0.75 * (1.0 - y_arr[i] / y_arr[0])
            ax.plot(x_arr[i:i+2], z_arr[i:i+2],
                    color=color, lw=lw, alpha=float(alpha),
                    solid_capstyle='round', ls=ls, zorder=4)
        # Endpoint at plate
        ax.scatter(x_arr[-1], z_arr[-1], s=90, color=color,
                   zorder=7, edgecolors='white', linewidths=1.2)
        ax.annotate(label, (x_arr[-1], z_arr[-1]),
                    xytext=(7, 0), textcoords='offset points',
                    fontsize=8, color=color, fontweight='bold')
        # Decision-point marker (~23 ft from plate)
        dec_idx = int(np.argmin(np.abs(y_arr - 23)))
        ax.scatter(x_arr[dec_idx], z_arr[dec_idx], s=40, color=color,
                   zorder=6, marker='D', edgecolors='white', linewidths=0.8,
                   alpha=0.7)

    _draw(x_s, z_s, y_s, s['color'],      f'{s["short"]}: {_ab(sp_name)}', 2.5, '-')
    _draw(x_c, z_c, y_c, CLOSER['color'], f'{cname}: {_ab(cp_name)}',      2.5, '--')

    # Strike zone outline (average strike zone ~1.6 ft wide, 1.6–3.4 ft tall)
    zone_rect = mpatches.Rectangle(
        (-0.83, 1.6), 1.66, 1.8,
        lw=1.2, edgecolor=MUTED, facecolor='none', ls=':', zorder=2)
    ax.add_patch(zone_rect)
    ax.text(0.0, 1.5, 'SZ', ha='center', va='top', fontsize=7,
            color=MUTED, zorder=3)

    # Legend for decision-point markers
    ax.scatter([], [], s=30, color=MUTED, marker='D',
               label='Decision point (~23 ft)', alpha=0.7)
    ax.legend(fontsize=7, facecolor=PANEL, edgecolor=GRIDC,
              labelcolor=FG, loc='upper right')

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(0.5, 7.5)
    ax.grid(color=GRIDC, lw=0.4, alpha=0.5, zorder=0)
    ax.set_xlabel('Horizontal, ft  (catcher\'s view)', color=MUTED, fontsize=8)
    ax.set_ylabel('Height, ft', color=MUTED, fontsize=8)
    ax.set_title(
        f'{tag}  (sim = {sim_val:.2f})\n'
        f'{s["short"]} {_ab(sp_name)}  →  {cname} {_ab(cp_name)}',
        color=FG, fontsize=9, fontweight='bold', pad=6,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — APPROACH & SEQUENCING
# ═══════════════════════════════════════════════════════════════════════════════

def build_fig3(starters, fp_starters, fp_closer):
    """
    One row per starter, three panels each:
      Col 0 — Count-based pitch selection: side-by-side stacked bars showing
               the starter vs Chapman in First Pitch, 0-2 Ahead, 3-ball Behind,
               and Overall usage.
      Col 1 — Starter's pitch transition matrix (what follows what).
      Col 2 — Chapman's pitch transition matrix (same layout for direct compare).
    """
    cname = fp_closer['name'].split(',')[0]
    n_s   = len(starters)

    fig = plt.figure(figsize=(22, 6 * n_s + 1.2), facecolor=BG)
    fig.text(0.5, 0.998,
             f'{TEAM_LABEL} {SEASON}  ·  Approach & Sequencing Analysis',
             ha='center', va='top', fontsize=18, fontweight='bold',
             color=FG, fontfamily='monospace')
    fig.text(0.5, 0.984,
             f'First-pitch tendencies, count-based pitch selection, '
             f'and what-follows-what for each starter vs {cname}',
             ha='center', va='top', fontsize=10, color=MUTED)

    outer = gridspec.GridSpec(
        n_s, 1, figure=fig,
        top=0.97, bottom=0.03, left=0.04, right=0.97, hspace=0.55,
    )

    for ri, s in enumerate(starters):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[ri], wspace=0.30)
        fp_s  = fp_starters[s['id']]

        _f3_count_approach(fig.add_subplot(inner[0]), s, fp_s, fp_closer, cname)
        _f3_transition    (fig.add_subplot(inner[1]), s, fp_s,
                           title=f'{s["short"]} Transition Matrix',
                           edge_color=s['color'])
        _f3_transition    (fig.add_subplot(inner[2]), s, fp_closer,
                           title=f'{cname} Transition Matrix',
                           edge_color=CLOSER['color'])

    out = os.path.join(DATA_DIR, f'boston_{SEASON}_sequencing.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  Figure 3 saved → {out}')


def _f3_count_approach(ax, s, fp_s, fp_c, cname):
    """
    Horizontal stacked bars comparing pitch selection by count situation.
    Each pair of bars = one count situation (starter on top, closer on bottom).
    """
    _sa(ax, grid=False)
    ax.spines['left'].set_visible(False)

    situations = list(fp_s['count_mixes'].keys())   # ['First Pitch', ...]
    n_sit = len(situations)

    # y positions: two bars per situation (starter above, closer below),
    # with a gap between situations
    bar_h   = 0.35
    gap     = 0.25
    group_h = 2 * bar_h + gap
    y_starts = [ri * (group_h + 0.25) for ri in range(n_sit)]

    # Collect all pitch types that appear anywhere
    all_pitches_ordered = []
    seen = set()
    for mix in list(fp_s['count_mixes'].values()) + list(fp_c['count_mixes'].values()):
        for p in mix.sort_values(ascending=False).index:
            if p not in seen:
                all_pitches_ordered.append(p)
                seen.add(p)

    for si, sit_name in enumerate(situations):
        y_top = y_starts[si] + bar_h + gap / 2    # starter bar y
        y_bot = y_starts[si] + gap / 2            # closer bar y

        for y_pos, fp, label, border in [
            (y_top, fp_s, s['short'], s['color']),
            (y_bot, fp_c, cname,      CLOSER['color']),
        ]:
            mix  = fp['count_mixes'].get(sit_name, pd.Series(dtype=float))
            left = 0.0
            for pitch in all_pitches_ordered:
                pct = float(mix.get(pitch, 0.0))
                if pct < 0.5:
                    continue
                ax.barh(y_pos, pct, bar_h, left=left,
                        color=_pc(pitch), edgecolor=BG, linewidth=0.4,
                        alpha=0.90)
                if pct > 8:
                    ax.text(left + pct / 2, y_pos, f'{_ab(pitch)}\n{pct:.0f}%',
                            ha='center', va='center', fontsize=6.5,
                            color='white', fontweight='bold')
                left += pct

            # Label on the left
            ax.text(-2, y_pos, label, ha='right', va='center',
                    color=border, fontsize=7.5, fontweight='bold')

        # Situation label between the two bars
        ax.text(50, y_starts[si] + bar_h / 2 + gap / 2,
                sit_name, ha='center', va='center',
                color=MUTED, fontsize=7.5, style='italic')

    ax.set_xlim(-18, 102)
    ax.set_ylim(-0.15, y_starts[-1] + group_h + 0.1)
    ax.set_yticks([])
    ax.set_xlabel('Usage %', color=MUTED, fontsize=8)
    ax.set_title(
        f'{s["short"]}  vs  {cname}\nPitch Selection by Count Situation',
        color=FG, fontsize=9, fontweight='bold', pad=6,
    )
    ax.grid(axis='x', color=GRIDC, linewidth=0.5, alpha=0.7)

    # Pitch type legend
    patches = [mpatches.Patch(color=_pc(p), label=_ab(p))
               for p in all_pitches_ordered if p in PITCH_COLORS]
    ax.legend(handles=patches, fontsize=6.5, facecolor=PANEL,
              edgecolor=GRIDC, labelcolor=FG,
              loc='lower right', ncol=2)


def _f3_transition(ax, s, fp, title, edge_color):
    """
    Heatmap of the pitch transition matrix.
    Row = pitch just thrown.  Column = next pitch thrown.
    Cell = probability (0–1) of that transition.
    """
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(GRIDC)
    ax.tick_params(colors=MUTED, labelsize=8)

    tm = fp.get('transition_matrix', pd.DataFrame())
    if tm.empty:
        ax.text(0.5, 0.5, 'Insufficient data\n(< 30 transitions)',
                ha='center', va='center', color=MUTED,
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title, color=FG, fontsize=9, fontweight='bold', pad=6)
        return

    # Abbreviate labels and sort by overall usage (most common first)
    pitches  = sorted(set(tm.index) | set(tm.columns))
    usage    = fp['pitch_mix']
    pitches  = sorted(pitches,
                      key=lambda p: -float(usage.get(p, 0)))
    tm_sq    = tm.reindex(index=pitches, columns=pitches, fill_value=0.0)

    mat      = tm_sq.values
    row_lbls = [_ab(p) for p in pitches]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'tr', [BG, PANEL, edge_color], N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(row_lbls)))
    ax.set_xticklabels(row_lbls, color=FG, fontsize=8.5)
    ax.set_yticks(range(len(row_lbls)))
    ax.set_yticklabels(row_lbls, color=FG, fontsize=8.5)

    for ri in range(len(pitches)):
        for ci in range(len(pitches)):
            v = mat[ri, ci]
            if v > 0.03:
                ax.text(ci, ri, f'{v:.0%}', ha='center', va='center',
                        fontsize=7.5,
                        color='white' if v > 0.35 else FG,
                        fontweight='bold' if v > 0.30 else 'normal')

    ax.set_xlabel('Next pitch →', color=MUTED, fontsize=8)
    ax.set_ylabel('← Prev pitch', color=MUTED, fontsize=8)
    ax.set_title(title, color=FG, fontsize=9, fontweight='bold', pad=6)

    # Highlight the dominant sequence per row
    for ri in range(len(pitches)):
        ci = int(np.argmax(mat[ri]))
        if mat[ri, ci] > 0.30:
            rect = mpatches.FancyBboxPatch(
                (ci - 0.48, ri - 0.48), 0.96, 0.96,
                boxstyle='round,pad=0.05',
                edgecolor='white', facecolor='none',
                linewidth=1.5, zorder=5)
            ax.add_patch(rect)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f'\n  ── {TEAM_LABEL} {SEASON} · Rotation → Closer Disorientation ──')

    # ── load ──────────────────────────────────────────────────────────────────
    print('  Loading data...')
    df_closer   = load_pitcher(CLOSER['id'], CLOSER.get('team_filter'))
    df_starters = {s['id']: load_pitcher(s['id'], s.get('team_filter'))
                   for s in STARTERS}

    # ── fingerprint ───────────────────────────────────────────────────────────
    print('  Building fingerprints...')
    fp_closer   = build_fingerprint(df_closer)
    fp_starters = {s['id']: build_fingerprint(df_starters[s['id']])
                   for s in STARTERS}

    # ── disorientation ────────────────────────────────────────────────────────
    nw = normalize_weights(WEIGHTS)
    print('  Computing disorientation scores...')
    dis = {s['id']: compute_disorientation(fp_starters[s['id']], fp_closer, nw)
           for s in STARTERS}

    # ── game linking ──────────────────────────────────────────────────────────
    print('  Linking games...')
    game_map = link_games(df_closer, df_starters)

    # ── per-game metrics ──────────────────────────────────────────────────────
    print('  Computing per-game closer metrics...')
    records = []
    for gpk, sid in game_map.items():
        g  = df_closer[df_closer['game_pk'] == gpk]
        m  = compute_game_metrics(g)
        st = next(s for s in STARTERS if s['id'] == sid)
        records.append({
            'game_pk':       gpk,
            'game_date':     g['game_date'].iloc[0],
            'starter_id':    sid,
            'starter_short': st['short'],
            'disorientation': dis[sid]['composite'],
            **m,
        })

    results_df = (pd.DataFrame(records)
                  .sort_values('game_date')
                  .reset_index(drop=True))

    total_matched = len(results_df)
    total_games   = df_closer['game_pk'].nunique()
    print(f'  Matched {total_matched} / {total_games} Chapman appearances '
          f'to a starter in this dataset.')
    if HAS_PYBASEBALL:
        print('  (pybaseball available — could extend with game-log ERA / FIP)')

    # ── summary ───────────────────────────────────────────────────────────────
    print_summary(results_df, dis, fp_starters, fp_closer, STARTERS)

    # ── visualize ─────────────────────────────────────────────────────────────
    print('  Generating figures...')
    build_fig1(results_df, dis, fp_starters, fp_closer,
               STARTERS, df_starters, df_closer)
    build_fig2(STARTERS, fp_starters, fp_closer)
    build_fig3(STARTERS, fp_starters, fp_closer)

    print(f'\n  Done.  Three PNGs saved to:\n  {DATA_DIR}')
    print('    boston_2025_dashboard.png')
    print('    boston_2025_pitches.png')
    print('    boston_2025_sequencing.png\n')


if __name__ == '__main__':
    main()
