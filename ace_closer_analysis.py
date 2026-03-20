
"""
Ace-Closer Disorientation Analysis
===================================
Quantifies how visually and mechanically disorienting it is for a batter
when a closer enters after the starting ace. Builds a fingerprint for each
pitcher across velocity, movement, release mechanics, spin, and pitch mix,
then scores the perceptual distance between them.

Usage:
    python3 ace_closer_analysis.py

Add more pairs by appending to the PAIRS list at the top.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')


# ─── CONFIGURATION ────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Add more pairs here as you collect CSVs — just drop {pitcher_id}_data.csv
# into the same folder and add an entry below.
PAIRS = [
    {
        'ace':    {'id': '676979', 'role': 'Ace (SP)'},
        'closer': {'id': '547973', 'role': 'Closer (RP)'},
        'team':   'Boston Red Sox',
        'season': 2025,
    },
    # Example future pairs:
    # {'ace': {'id': '694973'}, 'closer': {'id': '605400'}, 'team': 'Pittsburgh Pirates', 'season': 2025},  # Skenes + Bednar
    # {'ace': {'id': '664285'}, 'closer': {'id': '607192'}, 'team': 'Houston Astros', 'season': 2025},     # Valdez + Clase
]

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
    'Other':           '#9E9E9E',
}

ACE_COLOR    = '#4FC3F7'   # cool blue
CLOSER_COLOR = '#FF7043'   # ember orange

BG_COLOR   = '#0D1117'
PANEL_BG   = '#161B22'
TEXT_COLOR = '#E6EDF3'
MUTED      = '#8B949E'
GRID_COLOR = '#30363D'

# ─── DATA LOADING ─────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    'release_speed', 'effective_speed', 'release_pos_x', 'release_pos_z',
    'release_extension', 'release_spin_rate', 'spin_axis', 'arm_angle',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
    'api_break_z_with_gravity', 'api_break_x_batter_in', 'api_break_x_arm',
]


def load_pitcher(pitcher_id: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f'{pitcher_id}_data.csv')
    df = pd.read_csv(path, low_memory=False)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Drop rows with no pitch name
    df = df[df['pitch_name'].notna() & (df['pitch_name'].str.strip() != '')]
    df['pitch_name'] = df['pitch_name'].str.strip()
    return df


# ─── PITCHER FINGERPRINTING ───────────────────────────────────────────────────

def build_fingerprint(df: pd.DataFrame) -> dict:
    """
    Distill a full season's worth of pitch data into a pitcher fingerprint.
    Returns overall mechanics, pitch mix, and per-pitch-type breakdowns.
    """
    name      = df['player_name'].iloc[0]
    handedness = df['p_throws'].mode()[0]
    total     = len(df)

    fp = {
        'name':          name,
        'hand':          handedness,
        'total_pitches': total,
        # Overall mechanics (median is robust to outliers)
        'arm_angle':      df['arm_angle'].median(),
        'release_z':      df['release_pos_z'].median(),
        'release_x':      df['release_pos_x'].median(),
        'extension':      df['release_extension'].median(),
        'velo':           df['release_speed'].median(),
        'effective_velo': df['effective_speed'].median(),
        'spin_rate':      df['release_spin_rate'].median(),
        'spin_axis':      df['spin_axis'].median(),
        # Weighted mean movement (feet; pitcher's POV)
        'pfx_x':          df['pfx_x'].median(),
        'pfx_z':          df['pfx_z'].median(),
        # Zone tendency
        'plate_x':        df['plate_x'].median(),
        'plate_z':        df['plate_z'].median(),
    }

    # Pitch mix (%)
    fp['pitch_mix'] = (df['pitch_name'].value_counts(normalize=True) * 100).round(2)

    # Per-pitch-type stats (only for pitch types with ≥ 15 pitches)
    by_pitch = {}
    for pitch, grp in df.groupby('pitch_name'):
        if len(grp) < 15:
            continue
        by_pitch[pitch] = {
            'count':    len(grp),
            'pct':      100 * len(grp) / total,
            'velo':     grp['release_speed'].mean(),
            'spin':     grp['release_spin_rate'].mean(),
            'spin_axis':grp['spin_axis'].mean(),
            # pfx in inches for readability
            'pfx_x_in': grp['pfx_x'].mean() * 12,
            'pfx_z_in': grp['pfx_z'].mean() * 12,
            'break_h':  grp['api_break_x_batter_in'].mean(),
            'break_v':  grp['api_break_z_with_gravity'].mean(),
            'plate_x':  grp['plate_x'].mean(),
            'plate_z':  grp['plate_z'].mean(),
        }
    fp['by_pitch'] = by_pitch

    return fp


def print_fingerprint(fp: dict, role: str):
    print(f"\n{'─'*60}")
    print(f"  {fp['name']}  [{role}]  — {fp['hand']}HP  — {fp['total_pitches']} pitches")
    print(f"{'─'*60}")
    print(f"  Velocity (median):  {fp['velo']:.1f} mph  (eff. {fp['effective_velo']:.1f})")
    print(f"  Extension:          {fp['extension']:.2f} ft")
    print(f"  Arm Angle:          {fp['arm_angle']:.1f}°")
    print(f"  Release (x, z):     ({fp['release_x']:.2f}, {fp['release_z']:.2f}) ft")
    print(f"  Spin Rate:          {fp['spin_rate']:.0f} rpm  (axis {fp['spin_axis']:.0f}°)")
    print(f"  Movement (pfx):     H {fp['pfx_x']*12:.1f}\" / V {fp['pfx_z']*12:.1f}\"")
    print(f"  Pitch Mix:")
    for pitch, pct in fp['pitch_mix'].items():
        bar = '█' * int(pct / 3)
        print(f"    {pitch:22s}  {pct:5.1f}%  {bar}")


# ─── DISORIENTATION SCORING ───────────────────────────────────────────────────

# Expected ranges for normalization (based on MLB population)
NORM_RANGES = {
    'arm_angle':      (0,    80),    # degrees
    'release_z':      (4.0,  7.5),   # feet (height)
    'release_x':      (-3.0, 3.0),   # feet (side)
    'extension':      (4.5,  7.5),   # feet
    'effective_velo': (83,   105),    # mph
    'spin_axis':      (0,    360),    # degrees (handled circularly)
    'mix_js':         (0,    1),      # Jensen-Shannon divergence
    'movement':       (0,    20),     # inches (movement centroid distance)
}

SCORE_WEIGHTS = {
    'Handedness Flip':   0.15,
    'Arm Angle':         0.15,
    'Release Point':     0.12,
    'Effective Velocity':0.13,
    'Extension':         0.07,
    'Pitch Mix':         0.18,
    'Movement Profile':  0.12,
    'Spin Axis':         0.08,
}
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1"


def _norm(val, lo, hi):
    return float(np.clip((val - lo) / (hi - lo), 0, 1))


def compute_disorientation(fp1: dict, fp2: dict) -> dict:
    """
    Score the perceptual distance between two pitchers on each dimension.
    Returns sub-scores (0-1), weights, and a weighted composite.
    """
    raw = {}

    # 1. Handedness flip — biggest single visual shift
    raw['Handedness Flip'] = 0.0 if fp1['hand'] == fp2['hand'] else 1.0

    # 2. Arm angle — changes the visual plane the ball comes from
    arm_delta = abs(fp1['arm_angle'] - fp2['arm_angle'])
    raw['Arm Angle'] = _norm(arm_delta, 0, 40)

    # 3. Release point — Euclidean distance in the x-z release window
    rel_delta = np.sqrt(
        (fp1['release_x'] - fp2['release_x'])**2 +
        (fp1['release_z'] - fp2['release_z'])**2
    )
    raw['Release Point'] = _norm(rel_delta, 0, 2.0)

    # 4. Effective velocity — accounts for extension; what the batter actually "feels"
    velo_delta = abs(fp1['effective_velo'] - fp2['effective_velo'])
    raw['Effective Velocity'] = _norm(velo_delta, 0, 12.0)

    # 5. Extension — affects perceived velocity and tunnel window timing
    ext_delta = abs(fp1['extension'] - fp2['extension'])
    raw['Extension'] = _norm(ext_delta, 0, 2.0)

    # 6. Pitch mix — Jensen-Shannon divergence between usage distributions
    all_pitches = sorted(set(fp1['pitch_mix'].index) | set(fp2['pitch_mix'].index))
    p1 = np.array([fp1['pitch_mix'].get(p, 0.0) for p in all_pitches], dtype=float)
    p2 = np.array([fp2['pitch_mix'].get(p, 0.0) for p in all_pitches], dtype=float)
    p1 /= p1.sum()
    p2 /= p2.sum()
    raw['Pitch Mix'] = float(jensenshannon(p1, p2))  # already 0–1

    # 7. Movement profile — distance between usage-weighted movement centroids (inches)
    def movement_centroid(fp):
        pitches = fp['by_pitch']
        total_w = sum(v['pct'] for v in pitches.values())
        if total_w == 0:
            return np.array([0.0, 0.0])
        wx = sum(v['pct'] * v['pfx_x_in'] for v in pitches.values()) / total_w
        wz = sum(v['pct'] * v['pfx_z_in'] for v in pitches.values()) / total_w
        return np.array([wx, wz])

    m1 = movement_centroid(fp1)
    m2 = movement_centroid(fp2)
    move_dist = float(np.linalg.norm(m1 - m2))
    raw['Movement Profile'] = _norm(move_dist, 0, 20.0)

    # 8. Spin axis — circular distance (affects perceived trajectory shape)
    axis_delta = abs(fp1['spin_axis'] - fp2['spin_axis'])
    axis_delta = min(axis_delta, 360 - axis_delta)  # circular
    raw['Spin Axis'] = _norm(axis_delta, 0, 90)

    composite = sum(raw[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)

    return {
        'dimensions': raw,
        'weights':    SCORE_WEIGHTS,
        'composite':  composite,
    }


def print_disorientation(dis: dict):
    print(f"\n{'─'*60}")
    print(f"  DISORIENTATION SCORES")
    print(f"{'─'*60}")
    for dim in SCORE_WEIGHTS:
        score  = dis['dimensions'][dim]
        weight = dis['weights'][dim]
        bar_len = int(score * 25)
        bar = '█' * bar_len + '░' * (25 - bar_len)
        level = 'HIGH' if score >= 0.65 else ('MED' if score >= 0.35 else 'LOW')
        print(f"  {dim:22s}  {bar}  {score:.2f}  [{level}]  (w={weight:.2f})")
    print(f"{'─'*60}")
    print(f"  {'COMPOSITE':22s}  {'▓' * int(dis['composite']*25):<25s}  {dis['composite']:.3f}")


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.8)


def _pitch_color(name):
    return PITCH_COLORS.get(name, PITCH_COLORS['Other'])


def _abbrev(name):
    abbrevs = {
        '4-Seam Fastball': '4S',
        'Sinker':          'SI',
        'Cutter':          'CT',
        'Sweeper':         'SW',
        'Slider':          'SL',
        'Changeup':        'CH',
        'Split-Finger':    'SPL',
        'Curveball':       'CB',
        'Knuckle Curve':   'KC',
    }
    return abbrevs.get(name, name[:3])


# ── Panel 1: Release point scatter ──────────────────────────────────────────

def _plot_release(ax, df_ace, df_closer, fp_ace, fp_closer, ace_lbl, closer_lbl):
    _style_ax(ax)

    # Scatter raw pitches at low alpha to show density
    ax.scatter(df_ace['release_pos_x'], df_ace['release_pos_z'],
               s=6, color=ACE_COLOR, alpha=0.12, zorder=2)
    ax.scatter(df_closer['release_pos_x'], df_closer['release_pos_z'],
               s=6, color=CLOSER_COLOR, alpha=0.12, zorder=2)

    # Median centroid
    ax.scatter(fp_ace['release_x'], fp_ace['release_z'],
               s=160, color=ACE_COLOR, zorder=5, marker='o',
               edgecolors='white', linewidths=1.5)
    ax.scatter(fp_closer['release_x'], fp_closer['release_z'],
               s=160, color=CLOSER_COLOR, zorder=5, marker='D',
               edgecolors='white', linewidths=1.5)

    # Dashed line connecting centroids
    ax.plot([fp_ace['release_x'], fp_closer['release_x']],
            [fp_ace['release_z'], fp_closer['release_z']],
            color='white', alpha=0.35, linewidth=1.5, linestyle='--', zorder=3)

    # Labels
    x_off_ace = 0.12 if fp_ace['release_x'] < fp_closer['release_x'] else -0.12
    x_off_clo = -x_off_ace
    ax.annotate(f"{ace_lbl}\nArm: {fp_ace['arm_angle']:.0f}°",
                xy=(fp_ace['release_x'], fp_ace['release_z']),
                xytext=(fp_ace['release_x'] + x_off_ace, fp_ace['release_z'] + 0.12),
                color=ACE_COLOR, fontsize=7.5, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=ACE_COLOR, alpha=0.4))
    ax.annotate(f"{closer_lbl}\nArm: {fp_closer['arm_angle']:.0f}°",
                xy=(fp_closer['release_x'], fp_closer['release_z']),
                xytext=(fp_closer['release_x'] + x_off_clo, fp_closer['release_z'] - 0.2),
                color=CLOSER_COLOR, fontsize=7.5, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=CLOSER_COLOR, alpha=0.4))

    ax.set_xlabel('Horizontal Position (ft, catcher\'s view)', color=MUTED, fontsize=8)
    ax.set_ylabel('Release Height (ft)', color=MUTED, fontsize=8)
    ax.set_title('Release Window', color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=8)

    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACE_COLOR,    markersize=8, label=ace_lbl),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=CLOSER_COLOR, markersize=8, label=closer_lbl),
    ]
    ax.legend(handles=legend_els, fontsize=8, facecolor=PANEL_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR, loc='lower right')


# ── Panel 2: Movement chart ──────────────────────────────────────────────────

def _plot_movement(ax, fp_ace, fp_closer, ace_lbl, closer_lbl):
    _style_ax(ax)

    def draw_pitches(fp, edge_color, marker):
        for pitch, s in fp['by_pitch'].items():
            size = max(50, s['pct'] * 9)
            ax.scatter(s['pfx_x_in'], s['pfx_z_in'],
                       s=size, color=_pitch_color(pitch),
                       edgecolors=edge_color, linewidths=1.8,
                       marker=marker, zorder=5, alpha=0.92)
            ax.annotate(f"{marker_prefix(marker)}{_abbrev(pitch)}",
                        xy=(s['pfx_x_in'], s['pfx_z_in']),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color=edge_color, fontweight='bold')

    def marker_prefix(m):
        return '◆ ' if m == 'D' else '● '

    draw_pitches(fp_ace,    ACE_COLOR,    'o')
    draw_pitches(fp_closer, CLOSER_COLOR, 'D')

    ax.axhline(0, color=GRID_COLOR, linewidth=1.0, zorder=1)
    ax.axvline(0, color=GRID_COLOR, linewidth=1.0, zorder=1)

    ax.set_xlabel('Horizontal Break, in. (pitcher\'s POV → = arm-side)', color=MUTED, fontsize=8)
    ax.set_ylabel('Vertical Break, in. (vs. gravity)', color=MUTED, fontsize=8)
    ax.set_title('Pitch Movement Profile\n(dot size = usage %)', color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=8)

    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ACE_COLOR,    markersize=8, label=ace_lbl),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=CLOSER_COLOR, markersize=8, label=closer_lbl),
    ]
    ax.legend(handles=legend_els, fontsize=8, facecolor=PANEL_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)


# ── Panel 3: Radar / fingerprint ─────────────────────────────────────────────

RADAR_DIMS = [
    # (label, fp_key, lo, hi, higher_is_more)
    ('Velocity',     'velo',          84,  103,  True),
    ('Extension',    'extension',      4.5,  7.5, True),
    ('Arm Angle',    'arm_angle',      0,   80,   True),
    ('Spin Rate',    'spin_rate',   1700, 2800,   True),
    ('H. Movement',  'pfx_x',        -0.8,  1.2,  True),   # feet, raw direction
    ('V. Movement',  'pfx_z',        -0.2,  1.4,  True),
]


def _plot_radar(ax, fp_ace, fp_closer, ace_lbl, closer_lbl):
    n      = len(RADAR_DIMS)
    labels = [d[0] for d in RADAR_DIMS]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    def vals(fp):
        v = [_norm(fp[key], lo, hi) for _, key, lo, hi, _ in RADAR_DIMS]
        return v + v[:1]

    v_ace    = vals(fp_ace)
    v_closer = vals(fp_closer)

    ax.set_facecolor(PANEL_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, v_ace,    color=ACE_COLOR,    linewidth=2.0)
    ax.fill(angles, v_ace,    color=ACE_COLOR,    alpha=0.18)
    ax.plot(angles, v_closer, color=CLOSER_COLOR, linewidth=2.0)
    ax.fill(angles, v_closer, color=CLOSER_COLOR, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, color=TEXT_COLOR)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([], size=0)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.spines['polar'].set_color(GRID_COLOR)

    ax.set_title('Pitcher Fingerprint\n(radar overlap = similarity)',
                 color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=14)

    legend_els = [
        mpatches.Patch(color=ACE_COLOR,    alpha=0.7, label=ace_lbl),
        mpatches.Patch(color=CLOSER_COLOR, alpha=0.7, label=closer_lbl),
    ]
    ax.legend(handles=legend_els, loc='upper right', bbox_to_anchor=(1.4, 1.15),
              fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)


# ── Panel 4: Pitch mix ────────────────────────────────────────────────────────

def _plot_mix(ax, fp_ace, fp_closer, ace_lbl, closer_lbl):
    _style_ax(ax)
    ax.spines['left'].set_visible(False)

    all_pitches = sorted(
        set(fp_ace['pitch_mix'].index) | set(fp_closer['pitch_mix'].index),
        key=lambda p: -(fp_ace['pitch_mix'].get(p, 0) + fp_closer['pitch_mix'].get(p, 0))
    )
    y_pos  = [1.2, 0.5]
    fps    = [fp_ace, fp_closer]
    colors = [ACE_COLOR, CLOSER_COLOR]
    labels = [ace_lbl, closer_lbl]

    for yi, (fp, bar_color, lbl) in enumerate(zip(fps, colors, labels)):
        left = 0
        for pitch in all_pitches:
            pct = fp['pitch_mix'].get(pitch, 0)
            if pct < 0.5:
                continue
            ax.barh(y_pos[yi], pct, left=left, height=0.5,
                    color=_pitch_color(pitch), edgecolor=BG_COLOR, linewidth=0.6)
            if pct > 6:
                ax.text(left + pct / 2, y_pos[yi],
                        f'{_abbrev(pitch)}\n{pct:.0f}%',
                        ha='center', va='center', fontsize=7,
                        color='white', fontweight='bold')
            left += pct
        ax.text(-1.5, y_pos[yi], lbl, ha='right', va='center',
                color=bar_color, fontsize=9, fontweight='bold')

    # Pitch legend
    shown = set(all_pitches)
    patch_list = [mpatches.Patch(color=_pitch_color(p), label=p)
                  for p in PITCH_COLORS if p in shown]
    ax.legend(handles=patch_list, fontsize=6.5, facecolor=PANEL_BG,
              edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR,
              loc='lower right', ncol=2)

    ax.set_xlim(-2, 102)
    ax.set_ylim(0.1, 1.7)
    ax.set_yticks([])
    ax.set_xlabel('Usage %', color=MUTED, fontsize=8)
    ax.set_title('Pitch Mix', color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=8)
    ax.grid(axis='x', color=GRID_COLOR, linewidth=0.5)
    ax.grid(axis='y', visible=False)


# ── Panel 5: Key mechanics table ─────────────────────────────────────────────

def _plot_stats(ax, fp_ace, fp_closer, ace_lbl, closer_lbl):
    _style_ax(ax)
    ax.set_axis_off()
    ax.set_title('Mechanics at a Glance', color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=8)

    rows = [
        ('Avg Velocity',    f'{fp_ace["velo"]:.1f} mph',           f'{fp_closer["velo"]:.1f} mph'),
        ('Eff. Velocity',   f'{fp_ace["effective_velo"]:.1f} mph', f'{fp_closer["effective_velo"]:.1f} mph'),
        ('Extension',       f'{fp_ace["extension"]:.2f} ft',       f'{fp_closer["extension"]:.2f} ft'),
        ('Arm Angle',       f'{fp_ace["arm_angle"]:.1f}°',         f'{fp_closer["arm_angle"]:.1f}°'),
        ('Release Height',  f'{fp_ace["release_z"]:.2f} ft',       f'{fp_closer["release_z"]:.2f} ft'),
        ('Release Side',    f'{fp_ace["release_x"]:.2f} ft',       f'{fp_closer["release_x"]:.2f} ft'),
        ('Spin Rate',       f'{fp_ace["spin_rate"]:.0f} rpm',      f'{fp_closer["spin_rate"]:.0f} rpm'),
        ('Spin Axis',       f'{fp_ace["spin_axis"]:.0f}°',         f'{fp_closer["spin_axis"]:.0f}°'),
        ('H. Break (med)',  f'{fp_ace["pfx_x"]*12:.1f}"',          f'{fp_closer["pfx_x"]*12:.1f}"'),
        ('V. Break (med)',  f'{fp_ace["pfx_z"]*12:.1f}"',          f'{fp_closer["pfx_z"]*12:.1f}"'),
        ('Pitch Types',     str(len(fp_ace["by_pitch"])),          str(len(fp_closer["by_pitch"]))),
        ('Handedness',      f'{fp_ace["hand"]}HP',                  f'{fp_closer["hand"]}HP'),
    ]

    cx = [0.02, 0.47, 0.76]
    hy = 0.96

    ax.text(cx[0], hy, 'Stat',       transform=ax.transAxes, color=MUTED,        fontsize=8,   fontweight='bold')
    ax.text(cx[1], hy, ace_lbl,      transform=ax.transAxes, color=ACE_COLOR,    fontsize=8,   fontweight='bold')
    ax.text(cx[2], hy, closer_lbl,   transform=ax.transAxes, color=CLOSER_COLOR, fontsize=8,   fontweight='bold')

    line_y = hy - 0.04
    ax.plot([0, 1], [line_y, line_y], transform=ax.transAxes,
            color=GRID_COLOR, linewidth=0.8)

    rh = 0.076
    for i, (stat, v1, v2) in enumerate(rows):
        y = hy - 0.10 - i * rh
        if y < 0.02:
            break
        if i % 2 == 0:
            ax.add_patch(mpatches.FancyBboxPatch(
                (0, y - 0.01), 1, rh - 0.005,
                boxstyle='round,pad=0.005', facecolor='#1E2430',
                edgecolor='none', transform=ax.transAxes, zorder=0))
        ax.text(cx[0], y, stat, transform=ax.transAxes, color=MUTED,        fontsize=7.5)
        ax.text(cx[1], y, v1,   transform=ax.transAxes, color=ACE_COLOR,    fontsize=7.5, fontweight='bold')
        ax.text(cx[2], y, v2,   transform=ax.transAxes, color=CLOSER_COLOR, fontsize=7.5, fontweight='bold')


# ── Panel 6: Disorientation score breakdown ───────────────────────────────────

def _plot_disorientation(ax, dis: dict, ace_lbl: str, closer_lbl: str):
    _style_ax(ax)

    dims      = dis['dimensions']
    composite = dis['composite']

    labels = list(SCORE_WEIGHTS.keys())
    scores = [dims[k] for k in labels]

    bar_colors = []
    for s in scores:
        if s >= 0.65:
            bar_colors.append('#E63946')
        elif s >= 0.35:
            bar_colors.append('#F4A261')
        else:
            bar_colors.append('#4FC3F7')

    y_pos = list(range(len(labels)))
    bars  = ax.barh(y_pos, scores, color=bar_colors,
                    edgecolor=BG_COLOR, height=0.6, zorder=3)

    for bar, score in zip(bars, scores):
        ax.text(min(score + 0.03, 0.95),
                bar.get_y() + bar.get_height() / 2,
                f'{score:.2f}', va='center', fontsize=8,
                color=TEXT_COLOR, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5, color=TEXT_COLOR)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('Disorientation  (0 = identical  →  1 = maximally different)',
                  color=MUTED, fontsize=7.5)
    ax.set_title(
        f'Disorientation Breakdown\n'
        f'Composite Score: {composite:.3f} / 1.000',
        color=TEXT_COLOR, fontsize=10, fontweight='bold', pad=8)

    for xv, label in [(0.33, 'LOW'), (0.66, 'HIGH')]:
        ax.axvline(xv, color=GRID_COLOR, linewidth=1, linestyle='--', alpha=0.7, zorder=2)
        ax.text(xv, len(labels) - 0.1, label, fontsize=6.5,
                color=MUTED, ha='center', va='bottom')

    # Composite indicator line
    ax.axvline(composite, color='white', linewidth=1.5, linestyle='-', alpha=0.6, zorder=4)
    ax.text(composite, -0.7, f'▲ {composite:.2f}',
            fontsize=7.5, color='white', ha='center', fontweight='bold')

    # Weight annotations
    for i, dim in enumerate(labels):
        w = SCORE_WEIGHTS[dim]
        ax.text(1.01, i, f'w={w:.2f}', va='center', fontsize=6.5,
                color=MUTED, transform=ax.get_yaxis_transform())


# ── Master layout ─────────────────────────────────────────────────────────────

def plot_comparison(pair_cfg, df_ace, df_closer, fp_ace, fp_closer, dis):
    ace_lbl    = fp_ace['name'].split(',')[0].strip()
    closer_lbl = fp_closer['name'].split(',')[0].strip()
    team       = pair_cfg['team']
    season     = pair_cfg['season']

    fig = plt.figure(figsize=(22, 14), facecolor=BG_COLOR)

    # Title
    fig.text(0.5, 0.975,
             f'{ace_lbl}  →  {closer_lbl}   ·   {team}  {season}',
             ha='center', va='top', fontsize=22, fontweight='bold',
             color=TEXT_COLOR, fontfamily='monospace')
    fig.text(0.5, 0.948,
             'Ace-Closer Disorientation Analysis  ·  '
             'How different does a batter\'s world become when the closer enters?',
             ha='center', va='top', fontsize=10.5, color=MUTED)

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        left=0.05, right=0.97, top=0.93, bottom=0.06,
        hspace=0.38, wspace=0.30)

    ax_rel  = fig.add_subplot(gs[0, 0])
    ax_mov  = fig.add_subplot(gs[0, 1])
    ax_rad  = fig.add_subplot(gs[0, 2], projection='polar')
    ax_mix  = fig.add_subplot(gs[1, 0])
    ax_stat = fig.add_subplot(gs[1, 1])
    ax_dis  = fig.add_subplot(gs[1, 2])

    _plot_release     (ax_rel,  df_ace, df_closer, fp_ace, fp_closer, ace_lbl, closer_lbl)
    _plot_movement    (ax_mov,  fp_ace, fp_closer, ace_lbl, closer_lbl)
    _plot_radar       (ax_rad,  fp_ace, fp_closer, ace_lbl, closer_lbl)
    _plot_mix         (ax_mix,  fp_ace, fp_closer, ace_lbl, closer_lbl)
    _plot_stats       (ax_stat, fp_ace, fp_closer, ace_lbl, closer_lbl)
    _plot_disorientation(ax_dis, dis, ace_lbl, closer_lbl)

    fname = f'{ace_lbl.lower()}_{closer_lbl.lower()}_{season}_comparison.png'
    out   = os.path.join(DATA_DIR, fname)
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)
    print(f'\n  Saved → {out}')
    return out


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    for pair in PAIRS:
        ace_id    = pair['ace']['id']
        closer_id = pair['closer']['id']

        ace_csv    = os.path.join(DATA_DIR, f'{ace_id}_data.csv')
        closer_csv = os.path.join(DATA_DIR, f'{closer_id}_data.csv')

        if not os.path.exists(ace_csv):
            print(f'Missing: {ace_csv} — skipping pair.')
            continue
        if not os.path.exists(closer_csv):
            print(f'Missing: {closer_csv} — skipping pair.')
            continue

        print(f'\n{"═"*60}')
        print(f'  {pair["team"]}  {pair["season"]}')
        print(f'{"═"*60}')

        df_ace    = load_pitcher(ace_id)
        df_closer = load_pitcher(closer_id)

        fp_ace    = build_fingerprint(df_ace)
        fp_closer = build_fingerprint(df_closer)

        print_fingerprint(fp_ace,    pair['ace']['role'])
        print_fingerprint(fp_closer, pair['closer']['role'])

        dis = compute_disorientation(fp_ace, fp_closer)
        print_disorientation(dis)

        plot_comparison(pair, df_ace, df_closer, fp_ace, fp_closer, dis)


if __name__ == '__main__':
    main()
