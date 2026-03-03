"""
analysis/sentiment.py
Sentiment scoring from workout notes + correlation with performance.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter

GREEN  = '#22c55e'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'
BLUE   = '#4a9eed'

POSITIVE = [
    'fun','love','loved','great','strong','good','amazing','awesome',
    'happy','enjoy','enjoyed','solid','proud','unbroken','fast',
    'better','improved','survived','nailed','bright side','thank','excited','pr'
]
NEGATIVE = [
    'hard','struggle','struggled','hurt','pain','tired','fatigued',
    'slow','heavy','failed','miss','missed','sore','exhausted',
    'killer','rough','difficult','bad','weak','dying','worst',
    'challenging','humbling','sharp pain'
]


def score_sentiment(text):
    """Score a single note -1 (negative) to +1 (positive). None if no signal."""
    t   = str(text).lower()
    pos = sum(1 for w in POSITIVE if w in t)
    neg = sum(1 for w in NEGATIVE if w in t)
    if pos == 0 and neg == 0:
        return None
    return (pos - neg) / (pos + neg)


def enrich(df):
    """Add sentiment, pos_hits, neg_hits columns to df. Returns df."""
    df = df.copy()
    df['sentiment'] = df['notes'].apply(score_sentiment)
    df['pos_hits']  = df['notes'].str.lower().apply(
        lambda t: [w for w in POSITIVE if w in t])
    df['neg_hits']  = df['notes'].str.lower().apply(
        lambda t: [w for w in NEGATIVE if w in t])
    return df


def sentiment_over_time(df):
    """Monthly sentiment bar chart with trend line. Returns fig."""
    has_sent = df[df['sentiment'].notna()]
    monthly  = has_sent.groupby('month')['sentiment'].mean().reset_index()
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()
    monthly['rolling']  = monthly['sentiment'].rolling(3, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [GREEN if v >= 0 else RED for v in monthly['sentiment']]
    ax.bar(monthly['month_dt'], monthly['sentiment'], color=colors, alpha=0.7, width=20)
    ax.axhline(0, color='#999', linewidth=1)
    ax.plot(monthly['month_dt'], monthly['rolling'], color=PURPLE,
            linewidth=2.5, label='3-month trend')
    ax.set_title('Workout Sentiment Over Time', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Sentiment score  (positive ↑ / negative ↓)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    return fig


def word_frequency(df):
    """Top positive and negative words. Returns fig."""
    all_pos = [w for hits in df['pos_hits'] for w in hits]
    all_neg = [w for hits in df['neg_hits'] for w in hits]
    top_pos = Counter(all_pos).most_common(10)
    top_neg = Counter(all_neg).most_common(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    if top_pos:
        w, c = zip(*top_pos)
        ax1.barh(w, c, color=GREEN, alpha=0.8)
        ax1.set_title('Most Used Positive Words', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
    if top_neg:
        w, c = zip(*top_neg)
        ax2.barh(w, c, color=RED, alpha=0.8)
        ax2.set_title('Most Used Negative Words', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
    plt.tight_layout()
    return fig


def sentiment_vs_performance(df):
    """Dual-axis chart: sentiment bars + RX rate line. Returns (fig, r)."""
    corr_df = df[df['sentiment'].notna()].copy()
    corr_df['is_rx_num'] = (corr_df['rx_or_scaled'] == 'RX').astype(int)
    monthly  = corr_df.groupby('month').agg(
        avg_sentiment=('sentiment', 'mean'),
        rx_rate=('is_rx_num', 'mean')
    ).reset_index()
    months_dt = monthly['month'].dt.to_timestamp()
    r = monthly['avg_sentiment'].corr(monthly['rx_rate'])

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    ax1.bar(months_dt, monthly['avg_sentiment'], color=PURPLE, alpha=0.5, width=20, label='Sentiment')
    ax2.plot(months_dt, monthly['rx_rate'] * 100, color=GREEN, linewidth=2.5,
             marker='o', markersize=5, label='RX Rate %')
    ax1.set_ylabel('Avg Sentiment Score', color=PURPLE)
    ax2.set_ylabel('RX Rate (%)', color=GREEN)
    ax1.set_title('Does Mood Predict Performance?', fontsize=15, fontweight='bold', pad=15)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    return fig, r


def summary(df):
    """Return dict of key sentiment stats for agent use."""
    has_sent = df[df['sentiment'].notna()]
    top_pos  = has_sent.nlargest(3, 'sentiment')[['date','title','sentiment']].to_dict('records')
    top_neg  = has_sent.nsmallest(3, 'sentiment')[['date','title','sentiment']].to_dict('records')
    _, r = sentiment_vs_performance(df)
    plt.close('all')
    return {
        'notes_with_signal': len(has_sent),
        'avg_sentiment': round(has_sent['sentiment'].mean(), 2),
        'sentiment_vs_rx_correlation': round(r, 3),
        'most_positive_sessions': top_pos,
        'most_negative_sessions': top_neg,
    }
