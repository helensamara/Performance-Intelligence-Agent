"""
analysis/performance.py
Performance tracking — RX rate, strength progression, lift correlations,
scaling ratio, PR timeline.
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

BLUE   = '#4a9eed'
GREEN  = '#22c55e'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'


def rx_rate_over_time(df):
    """Monthly RX rate with 3-month trend. Returns fig."""
    monthly = df.groupby('month').apply(
        lambda x: (x['rx_or_scaled'] == 'RX').mean() * 100
    ).reset_index(name='rx_pct')
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()
    monthly['rolling']  = monthly['rx_pct'].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(monthly['month_dt'], monthly['rx_pct'], alpha=0.15, color=BLUE)
    ax.plot(monthly['month_dt'], monthly['rx_pct'], color=BLUE,
            linewidth=2, marker='o', markersize=5)
    ax.plot(monthly['month_dt'], monthly['rolling'], color=GREEN,
            linewidth=2.5, label='3-month trend')
    ax.axhline(50, color='#ccc', linestyle='--', linewidth=1, label='50% mark')
    ax.set_title('RX Rate Over Time', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('% of sessions RX')
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    return fig


def strength_progression(df, top_n=6):
    """Per-lift progression with running max and PR markers. Returns fig."""
    strength  = df[(df['score_type'] == 'Load') & (df['barbell_lift'] != '')].copy()
    strength['load'] = pd.to_numeric(strength['best_result_raw'], errors='coerce')
    top_lifts = strength['barbell_lift'].value_counts().head(top_n).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    for i, lift in enumerate(top_lifts):
        ax   = axes[i]
        data = strength[strength['barbell_lift'] == lift].sort_values('date').copy()
        prs  = data[data['is_pr']]
        ax.plot(data['date'], data['load'], 'o-', color=BLUE, alpha=0.5, markersize=4, linewidth=1)
        if len(prs):
            ax.scatter(prs['date'], prs['load'], color=AMBER, s=100, zorder=5, marker='*', label='PR')
        data['running_max'] = data['load'].cummax()
        ax.plot(data['date'], data['running_max'], color=GREEN,
                linewidth=2, linestyle='--', label='Best ever')
        ax.set_title(lift, fontsize=12, fontweight='bold')
        ax.set_ylabel('lbs')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        if len(prs):
            ax.legend(fontsize=9)
    for j in range(len(top_lifts), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Strength Progression by Lift', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def lift_correlation(df):
    """Heatmap of correlation between lifts. Returns fig."""
    strength = df[(df['score_type'] == 'Load') & (df['barbell_lift'] != '')].copy()
    strength['load'] = pd.to_numeric(strength['best_result_raw'], errors='coerce')
    pivot = strength.groupby(['date', 'barbell_lift'])['load'].max().unstack()
    corr  = pivot.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                ax=ax, square=True, linewidths=0.5,
                annot_kws={'size': 9}, vmin=-1, vmax=1)
    ax.set_title('Which Lifts Improve Together?', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig, corr


def scaling_ratio(df):
    """Actual vs prescribed weight ratio over time. Returns (fig, summary_dict)."""
    def _rx_weight(desc):
        m = re.findall(r'(\d+)\s*/\s*(\d+)\s*(?:lbs?|#)?', str(desc))
        return int(m[0][1]) if m else None

    def _actual_weight(notes):
        m = re.findall(r'(\d+)\s*(?:lbs?|#|lb)\b', str(notes).lower())
        weights = [int(x) for x in m if 5 <= int(x) <= 400]
        return weights[0] if weights else None

    scaled = df[df['rx_or_scaled'] == 'SCALED'].copy()
    scaled['rx_weight']     = scaled['description'].apply(_rx_weight)
    scaled['actual_weight'] = scaled['notes'].apply(_actual_weight)
    scaled['ratio']         = scaled['actual_weight'] / scaled['rx_weight']
    paired = scaled[scaled['ratio'].notna() & scaled['ratio'].between(0.1, 1.0)].copy()
    paired = paired.sort_values('date').reset_index(drop=True)

    z    = np.polyfit(range(len(paired)), paired['ratio'] * 100, 1)
    p_fn = np.poly1d(z)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(paired['date'], paired['ratio'] * 100, alpha=0.6, color=PURPLE, s=40)
    ax.plot(paired['date'], p_fn(range(len(paired))), color=AMBER,
            linewidth=2.5, label=f'Trend ({z[0]:+.2f}%/session)')
    ax.axhline(100, color=GREEN, linestyle='--', linewidth=1.5, label='RX (100%)')
    ax.set_title('Scaling Ratio — How Close to RX Weight?', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('% of prescribed weight used')
    ax.set_ylim(0, 115)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    stats = {
        'pairs_found': len(paired),
        'avg_ratio_pct': round(paired['ratio'].mean() * 100, 1),
        'trend_per_session': round(z[0], 4),
    }
    return fig, stats


def pr_timeline(df):
    """Timeline of all PRs. Returns fig."""
    prs = df[df['is_pr']].copy()
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.scatter(prs['date'], [1] * len(prs), s=120, color=AMBER, zorder=5, marker='*')
    for i, (_, row) in enumerate(prs.iterrows()):
        ax.annotate(row['title'][:18], (row['date'], 1),
                    xytext=(0, 14 if i % 2 == 0 else -20),
                    textcoords='offset points',
                    ha='center', fontsize=7.5, color='#555', rotation=35)
    ax.set_title(f'Your {len(prs)} PRs', fontsize=16, fontweight='bold', pad=15)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.set_ylim(0.75, 1.5)
    plt.tight_layout()
    return fig


def summary(df):
    """Return dict of key performance stats for agent use."""
    total_rx   = (df['rx_or_scaled'] == 'RX').sum()
    total_sc   = (df['rx_or_scaled'] == 'SCALED').sum()
    _, sc_stats = scaling_ratio(df)
    plt.close('all')
    return {
        'total_prs': int(df['is_pr'].sum()),
        'rx_sessions': int(total_rx),
        'scaled_sessions': int(total_sc),
        'rx_rate_pct': round(total_rx / len(df) * 100, 1),
        'scaling_ratio': sc_stats,
    }
