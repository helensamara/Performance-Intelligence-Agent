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
    strength['load'] = pd.to_numeric(strength['best_result_raw'], errors='coerce')
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

def forecast_prs(df, top_n=6, horizon_days=90):
    """
    Linear regression on each lift's running max curve.
    Returns (forecasts_dict, fig).
    """
    strength = df[(df['score_type'] == 'Load') & (df['barbell_lift'] != '')].copy()
    strength['load'] = pd.to_numeric(strength['best_result_raw'], errors='coerce')
    top_lifts = strength['barbell_lift'].value_counts().head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    forecasts = {}

    for i, lift in enumerate(top_lifts):
        ax   = axes[i]
        data = strength[strength['barbell_lift'] == lift].sort_values('date').copy()
        data['days']        = (data['date'] - data['date'].min()).dt.days
        data['running_max'] = data['load'].cummax()

        reg = LinearRegression().fit(
            data['days'].values.reshape(-1, 1),
            data['running_max'].values
        )
        last_day     = data['days'].max()
        future_days  = np.arange(last_day, last_day + horizon_days + 1).reshape(-1, 1)
        future_dates = pd.date_range(data['date'].max(), periods=horizon_days + 1, freq='D')
        y_future     = reg.predict(future_days)

        current_max = data['running_max'].max()
        next_pr_day = next(
            (int(fd - last_day) for fd, fv in zip(future_days.flatten(), y_future)
             if fv > current_max + 2.5),
            None
        )
        forecasts[lift] = {
            'current_max_lbs': current_max,
            f'predicted_{horizon_days}d_lbs': round(y_future[-1], 1),
            'days_to_next_pr': next_pr_day,
        }

        # Plot
        ax.plot(data['date'], data['load'], 'o', color=BLUE, alpha=0.4, markersize=3)
        ax.plot(data['date'], data['running_max'], color=GREEN, linewidth=2, label='Best ever')
        ax.plot(future_dates, y_future, color=AMBER, linewidth=2, linestyle='--', label='Forecast')
        prs_l = data[data['is_pr']]
        if len(prs_l):
            ax.scatter(prs_l['date'], prs_l['load'], color=AMBER, s=80, zorder=5, marker='*')
        ax.set_title(lift, fontsize=12, fontweight='bold')
        ax.set_ylabel('lbs')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.legend(fontsize=8)

    for j in range(len(top_lifts), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('PR Forecasting — Linear Regression on Lift Curves',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    return forecasts, fig
