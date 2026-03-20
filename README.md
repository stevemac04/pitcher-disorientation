# ⚾ Pitcher Disorientation & Closer Advantage Analysis

> **🚧 WORK IN PROGRESS 🚧**  
> This project is under active development. Scoring weights, normalization ranges, and methodology are still being refined. Findings are exploratory and should not be treated as final or publication-ready. Expect breaking changes.

---

## What Is This?

The core idea is simple: when a closer enters a game, batters don't get a clean slate. They've spent the last 5–6 innings calibrating their timing and recognition to the starter — his arm slot, his velocity, his pitch shapes, how he sequences. The closer disrupts all of that. But *how much* disruption depends entirely on how different that closer actually is.

This project tries to quantify that disorientation — building a **pitcher fingerprint** for every pitcher in a rotation and for the closer, then scoring the perceptual and mechanical distance between each pairing. It then cross-references those scores against the closer's actual performance on the nights each starter was on the bump.

**The working hypothesis:** closers facing lineups that are maximally mis-calibrated to their stuff should outperform their raw metrics alone would predict.

The 2025 Boston Red Sox (Crochet, Bello, Giolito, Buehler → Chapman) are the initial test case.

---

## Outputs

Running the script produces three figures:

### `boston_2025_dashboard.png`
Macro-level comparison across all starter–closer pairings:
- Release window scatter (all pitchers overlaid from catcher's view)
- Pitch movement cloud (all arsenals, dot size = usage %)
- Disorientation score matrix (heatmap, each starter × 11 dimensions + composite)
- Chapman's performance breakdown by starter (K%, Whiff%, Chase%, H/BF, wOBA)
- Disorientation vs K% scatter with regression line
- Disorientation vs wOBA against scatter with regression line

### `boston_2025_pitch_analysis.png`
Pitch-level deep dive for each starter:
- Pitch similarity heatmap (starter pitches × Chapman pitches — cross-family pairs dimmed and excluded from "most similar" selection)
- Most similar pitch pair trajectory — batter's-eye view of both pitches in flight, reconstructed from Statcast kinematic data
- Most dissimilar pitch pair trajectory — same view showing maximum visual contrast

### `boston_2025_sequencing.png`
Approach and sequencing comparison for each starter:
- Count-based pitch selection (first pitch, ahead 0-2/1-2, behind 2-0/3-1, overall)
- Starter's pitch transition matrix (what follows what within at-bats)
- Chapman's pitch transition matrix side-by-side for direct comparison

---

## Disorientation Dimensions

The composite disorientation score is built from 11 weighted dimensions across two categories:

**Mechanics** — what the batter sees coming off the mound
| Dimension | What It Captures |
|---|---|
| Handedness Flip | LHP vs RHP — biggest single visual shift |
| Arm Angle | Visual plane the ball comes from |
| Release Point | Where in the window the ball appears |
| Effective Velocity | Batter's timing calibration (accounts for extension) |
| Extension | How far toward the plate at release |
| Movement Profile | How differently the ball moves overall |
| Spin Axis | Perceived rotation and trajectory shape |

**Approach** — how the pitcher uses his stuff situationally
| Dimension | What It Captures |
|---|---|
| Arsenal Overlap | Which pitch types exist and in what proportions |
| First Pitch Pattern | How they open at-bats |
| Two-Strike Approach | What they throw when ahead 0-2, 1-2, 2-2 |
| Pitch Sequencing | What pitch follows what (transition matrix divergence) |

All weights are **user-adjustable** at the top of `rotation_closer_analysis.py` and auto-normalize to sum to 1. Set any weight to `0` to disable a dimension entirely.

---

## Setup

### Requirements
```
python 3.10+
pandas
numpy
matplotlib
scipy
```

Install dependencies:
```bash
pip install pandas numpy matplotlib scipy
```

Optional (not required to run):
```bash
pip install pybaseball   # flagged in script where it could extend analysis
```

### Getting Data

All pitch data is sourced from **[Baseball Savant](https://baseballsavant.mlb.com)** via MLB's **Statcast** tracking system. Statcast is MLB's ball- and player-tracking infrastructure, installed in all 30 major league stadiums, that captures pitch-by-pitch kinematics — velocity, spin rate, release point, movement, and more — for every pitch thrown in the big leagues.

To download data:
1. Go to [Baseball Savant Statcast Search](https://baseballsavant.mlb.com/statcast_search)
2. Set **Player Type = Pitcher**, select season, search by name
3. Hit **Download CSV**
4. Rename the file `{pitcher_id}_data.csv` and drop it in the project folder

The pitcher ID is in the URL of their Baseball Savant profile page. Files used for the BOS 2025 analysis:

| Pitcher | ID | Notes |
|---|---|---|
| Aroldis Chapman | `547973` | Closer |
| Garrett Crochet | `676979` | |
| Brayan Bello | `678394` | |
| Lucas Giolito | `608337` | |
| Walker Buehler | `621111` | Filtered to BOS games only (also threw for PHI) |

---

## Running It

```bash
cd path/to/baseball
python3 rotation_closer_analysis.py
```

The script will print a summary table to the console and save all three figures to the same folder.

To add more pitchers or change the team, edit the `CLOSER` and `STARTERS` lists at the top of `rotation_closer_analysis.py`. The `team_filter` field restricts a pitcher's data to games where they threw for a specific team — useful for mid-season trades like Buehler.

---

## Early Findings — BOS 2025 *(WIP)*

> Scores and metrics below reflect the current default weighting. Results will shift as weights and normalization are tuned.

Chapman matched to **51 of 67** appearances against a starter in this dataset.

| Starter | Hand | Dis. Score | n | K% | Whiff% | Chase% | H/BF | wOBA | Avg EV |
|---|---|---|---|---|---|---|---|---|---|
| Crochet | LHP | 0.411 | 12 | 32.1% | 28.5% | 38.0% | .100 | .153 | 82.1 |
| Buehler | RHP | 0.508 | 13 | 43.8% | 46.2% | 39.0% | .073 | .164 | 83.9 |
| Giolito | RHP | 0.542 | 13 | 36.9% | 32.1% | 34.4% | .077 | .111 | 82.7 |
| Bello | RHP | **0.645** | 13 | **46.8%** | 33.2% | **40.5%** | **.069** | **.081** | 83.3 |

The Crochet → Chapman pairing has the lowest disorientation of the group, which makes sense — both are LHP, both sit 94–98 mph, and their arm angle range overlaps. The gap is mostly in pitch mix and sequencing. The three RHP starters all produce more disorientation when Chapman comes in, with Bello generating the highest composite score. Chapman's best outings (lowest wOBA, highest K%) track with the higher-disorientation pairings — consistent with the hypothesis, though **n is small and we're still in exploratory territory**.

---

## Pitch Similarity Rules

A fastball can never be scored as "most similar" to a breaking ball regardless of how their metrics line up numerically. Pitch families are enforced:

- **Fastball family:** 4-Seam, Sinker, Cutter
- **Breaking family:** Slider, Sweeper, Curveball, Knuckle Curve
- **Offspeed family:** Changeup, Split-Finger, Splitter

Cross-family pairs are hard-capped at a similarity score of 0.30 and visually dimmed in the heatmap. The trajectory panels only draw "most similar" comparisons from within-family pairs.

---

## Roadmap

- [ ] Expand to additional MLB teams (especially high-contrast pairings like HOU Framber Valdez → Ryan Pressly, or any same-hand + opposite-hand rotation combos)
- [ ] Empirical normalization using league-wide distributions instead of hardcoded ranges
- [ ] Add batter handedness splits (L vs R batters may respond to disorientation differently)
- [ ] Incorporate `pybaseball` for game-log ERA/FIP to cross-validate per-game wOBA estimates
- [ ] Multi-season analysis to increase sample size and test whether disorientation effect is stable year-over-year
- [ ] Full Hawk-Eye / Statcast Complete dataset would unlock: actual optical-tracked pitch trajectories (vs. kinematic extrapolation), pitcher delivery timing as a fingerprint dimension, and batter-level same-game exposure records — turning aggregate correlations into individual at-bat probability models

---

## Data & Attribution

All pitch-by-pitch data is sourced from **[Baseball Savant](https://baseballsavant.mlb.com)**, MLB's public Statcast portal.  
Statcast data is property of MLB Advanced Media. This project uses only the publicly available CSV export for non-commercial, research purposes.  
No proprietary data or internal Hawk-Eye feeds are used.

---

## Files

```
├── rotation_closer_analysis.py   # main script — all analysis and visualization
├── ace_closer_analysis.py        # earlier two-pitcher comparison (deprecated)
├── README.md
├── .gitignore
├── boston_2025_dashboard.png     # sample output — macro view
├── boston_2025_pitch_analysis.png  # sample output — pitch similarity & trajectories
└── boston_2025_sequencing.png    # sample output — approach & sequencing
```

---

*Built with public Statcast data. Part of a broader exploration into pitcher fingerprinting, closer construction, and the mechanics of batter disorientation. Work in progress — methodology subject to change.*
