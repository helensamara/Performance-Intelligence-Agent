"""
analysis/ml_models.py
Machine learning layer — clustering, anomaly detection, PR forecasting.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

BLUE   = '#4a9eed'
GREEN  = '#22c55e'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'


# ── Clustering ────────────────────────────────────────────────────

def _build_features(df):
    c = df.copy()
    c['is_rx_num']        = (c['rx_or_scaled'] == 'RX').astype(int)
    c['is_pr_num']        = c['is_pr'].astype(int)
    c['is_strength']      = (c['score_type'] == 'Load').astype(int)
    c['is_timed']         = c['best_result_display'].str.contains(r'\d+:\d+', na=False).astype(int)
    c['is_amrap']         = (c['score_type'] == 'Rounds + Reps').astype(int)
    c['has_barbell']      = (c['barbell_lift'] != '').astype(int)
    c['sentiment_filled'] = c['sentiment'].fillna(0) if 'sentiment' in c.columns else 0
    return c


def _name_cluster(row):
    if row['strength_rate'] > 0.5: return 'Strength / Lifting'
    if row['amrap_rate']    > 0.3: return 'AMRAP Metcon'
    if row['timed_rate']    > 0.3: return 'Timed Metcon'
    return 'Accessory / Mixed'


def cluster_workouts(df, k=4):
    """
    KMeans clustering of workouts into archetypes.
    Returns (df_with_clusters, profiles_df, fig).
    """
    df = _build_features(df)
    features = ['is_rx_num', 'is_strength', 'is_timed', 'is_amrap',
                'has_barbell', 'sentiment_filled', 'is_pr_num']
    X = StandardScaler().fit_transform(df[features].fillna(0))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X)

    profiles = df.groupby('cluster').agg(
        count=('cluster', 'size'),
        rx_rate=('is_rx_num', 'mean'),
        strength_rate=('is_strength', 'mean'),
        timed_rate=('is_timed', 'mean'),
        amrap_rate=('is_amrap', 'mean'),
        pr_rate=('is_pr_num', 'mean'),
    ).round(2)
    profiles['name'] = profiles.apply(_name_cluster, axis=1)
    df['cluster_name'] = df['cluster'].map(profiles['name'])

    # Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df['cluster_name'].value_counts()
    colors = [BLUE, GREEN, AMBER, PURPLE][:len(counts)]
    bars   = ax.bar(counts.index, counts.values, color=colors, alpha=0.85)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha='center', fontsize=12, fontweight='bold')
    ax.set_title('Workout Archetypes — Auto-Clustered (KMeans)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Number of sessions')
    plt.tight_layout()

    return df, profiles, fig


# ── Anomaly Detection ─────────────────────────────────────────────

def detect_anomalies(df, contamination=0.08):
    """
    Isolation Forest on strength sessions.
    Returns (anomalies_df, fig).
    """
    strength = df[(df['score_type'] == 'Load') & (df['barbell_lift'] != '')].copy()
    strength['load'] = strength['score_load']
    strength = strength[strength['load'].notna()].sort_values('date').reset_index(drop=True)
    strength['day_num']    = strength['date'].dt.dayofweek
    strength['days_since'] = strength['date'].diff().dt.days.fillna(0)

    iso = IsolationForest(contamination=contamination, random_state=42)
    strength['anomaly'] = iso.fit_predict(
        strength[['load', 'day_num', 'days_since']].fillna(0)
    )

    anomalies = strength[strength['anomaly'] == -1]
    normals   = strength[strength['anomaly'] == 1]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(normals['date'],   normals['load'],   color=BLUE, alpha=0.5, s=30, label='Normal')
    ax.scatter(anomalies['date'], anomalies['load'], color=RED,  alpha=0.9, s=80,
               marker='X', zorder=5, label=f'Anomaly ({len(anomalies)} sessions)')
    ax.set_title('Anomaly Detection — Unusual Strength Sessions (Isolation Forest)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Load (lbs)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    return anomalies[['date', 'title', 'load', 'barbell_lift']], fig


# ── PR Forecasting ────────────────────────────────────────────────

# Config per score type: (score_col, group_col, higher_is_better, ylabel, pr_delta)
_FORECAST_CFG = {
    'Load':          ('score_load',      'barbell_lift', True,  'lbs',         2.5),
    'Rounds + Reps': ('best_result_raw', 'title',        True,  'rounds+reps', 0.05),
    'Reps':          ('score_reps',      'title',        True,  'reps',        1.0),
    'Time':          ('score_seconds',   'title',        False, 'seconds',     10.0),
}

_BLOCK_HALF_LIFE   = 180   # days — recency decay for weighted regression
_MIN_STAGNANT      = 6     # sessions without improvement → new block boundary
_MIN_BLOCK_SESSIONS = 4    # fall back to full history if block is smaller


def _detect_block_start(running_best, min_stagnant=_MIN_STAGNANT):
    """Return (block_start_iloc, is_plateau) for the most recent training block.

    A block boundary is a gap of >= min_stagnant consecutive sessions where the
    running best did not improve.  is_plateau is True when the tail of the series
    (last min_stagnant entries) shows no improvement.
    """
    rb = np.asarray(running_best, dtype=float)
    n  = len(rb)
    if n <= min_stagnant:
        return 0, False

    # Indices where the running best actually improved
    improving = np.where(np.diff(rb) > 0)[0] + 1

    # Plateau: no improvement in the last min_stagnant sessions
    tail_gap   = (n - 1 - improving[-1]) if len(improving) else n
    is_plateau = len(improving) == 0 or tail_gap >= min_stagnant

    if len(improving) == 0:
        return 0, is_plateau

    if is_plateau:
        # Current block = everything from the last improvement onward
        return int(improving[-1]), True

    # Improving: scan backwards for the gap that opened this block
    for i in range(len(improving) - 1, 0, -1):
        if improving[i] - improving[i - 1] >= min_stagnant:
            return int(improving[i]), False

    return 0, False  # whole history is one improvement block


def forecast_prs(df, top_n=6, horizon_days=90, score_type='Load'):
    """
    Block-aware PR forecasting with recency-weighted regression.

    For each workout, the model:
      1. Detects the current training block (improvement run or plateau phase).
      2. Fits a recency-weighted linear regression on that block only,
         so early history doesn't drag the slope down during active cycles
         and a plateau naturally produces a near-flat forecast.
      3. Falls back to full-history recency-weighted fit if the detected block
         is too small (< _MIN_BLOCK_SESSIONS).

    score_type: 'Load' (lbs, higher=better) | 'Rounds + Reps' (decimal, higher=better)
                'Reps' (count, higher=better) | 'Time' (seconds, lower=better)

    Returns (forecasts_dict, fig).
    """
    if score_type not in _FORECAST_CFG:
        raise ValueError(f'score_type must be one of {list(_FORECAST_CFG)}')

    score_col, group_col, higher_is_better, ylabel, pr_delta = _FORECAST_CFG[score_type]

    sessions = df[df['score_type'] == score_type].copy()
    if score_type == 'Load':
        sessions = sessions[sessions['barbell_lift'] != '']
    sessions['score'] = pd.to_numeric(sessions[score_col], errors='coerce')
    sessions = sessions[sessions['score'].notna()]

    top_groups = sessions[group_col].value_counts().head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    forecasts = {}

    for i, name in enumerate(top_groups):
        ax   = axes[i]
        data = sessions[sessions[group_col] == name].sort_values('date').copy()
        data['days']         = (data['date'] - data['date'].min()).dt.days
        data['running_best'] = (data['score'].cummax()
                                if higher_is_better else data['score'].cummin())

        # ── Block detection ──────────────────────────────────────────
        block_start, is_plateau = _detect_block_start(data['running_best'].values)
        block = (data.iloc[block_start:]
                 if len(data) - block_start >= _MIN_BLOCK_SESSIONS
                 else data)

        # ── Recency-weighted regression on current block ─────────────
        last_day  = int(data['days'].max())
        fit_days  = block['days'].values
        fit_rb    = block['running_best'].values
        weights   = np.exp(-np.log(2) * (last_day - fit_days) / _BLOCK_HALF_LIFE)

        reg = LinearRegression().fit(
            fit_days.reshape(-1, 1), fit_rb, sample_weight=weights
        )
        future_days  = np.arange(last_day, last_day + horizon_days + 1).reshape(-1, 1)
        future_dates = pd.date_range(data['date'].max(), periods=horizon_days + 1, freq='D')
        y_future     = reg.predict(future_days)

        # ── PR prediction ────────────────────────────────────────────
        current_best = float(data['running_best'].iloc[-1])
        if higher_is_better:
            next_pr_day = next(
                (int(fd - last_day) for fd, fv in zip(future_days.flatten(), y_future)
                 if fv > current_best + pr_delta),
                None,
            )
        else:
            next_pr_day = next(
                (int(fd - last_day) for fd, fv in zip(future_days.flatten(), y_future)
                 if fv < current_best - pr_delta),
                None,
            )

        forecasts[name] = {
            f'current_best_{ylabel}': round(current_best, 2),
            f'predicted_{horizon_days}d_{ylabel}': round(float(y_future[-1]), 2),
            'days_to_next_pr': next_pr_day,
            'phase': 'plateau' if is_plateau else 'improving',
        }

        # ── Chart ────────────────────────────────────────────────────
        forecast_color = RED if is_plateau else AMBER
        forecast_label = 'Forecast (plateau)' if is_plateau else 'Forecast'

        ax.plot(data['date'], data['score'], 'o', color=BLUE, alpha=0.4, markersize=3)
        ax.plot(data['date'], data['running_best'], color=GREEN, linewidth=2, label='Best ever')
        ax.plot(future_dates, y_future, color=forecast_color,
                linewidth=2, linestyle='--', label=forecast_label)
        if not is_plateau and block_start > 0:
            ax.axvline(data['date'].iloc[block_start], color='#aaa',
                       linestyle=':', linewidth=1, label='Block start')
        prs_l = data[data['is_pr']]
        if len(prs_l):
            ax.scatter(prs_l['date'], prs_l['score'], color=AMBER, s=80, zorder=5, marker='*')
        ax.set_title(str(name)[:25], fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.legend(fontsize=8)

    for j in range(len(top_groups), len(axes)):
        axes[j].set_visible(False)

    type_labels = {'Load': 'Lift', 'Rounds + Reps': 'AMRAP', 'Reps': 'Reps', 'Time': 'Timed Workout'}
    plt.suptitle(f'PR Forecasting — {type_labels.get(score_type, score_type)} Progressions',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    return forecasts, fig
