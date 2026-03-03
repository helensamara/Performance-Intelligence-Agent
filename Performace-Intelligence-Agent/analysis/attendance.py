"""
analysis/attendance.py
Attendance analysis — consistency, gaps, day-of-week patterns.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BLUE   = '#4a9eed'
GREEN  = '#22c55e'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'


def monthly_chart(df):
    """Bar chart of workouts per month. Returns (fig, summary_dict)."""
    monthly = df.groupby('month').size().reset_index(name='count')
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()
    best_pos  = int(monthly['count'].values.argmax())
    worst_pos = int(monthly['count'].values.argmin())
    avg = monthly['count'].mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(monthly['month_dt'], monthly['count'], color=BLUE, alpha=0.8, width=20)
    bars[best_pos].set_color(GREEN)
    bars[worst_pos].set_color(RED)
    ax.axhline(avg, color=AMBER, linestyle='--', linewidth=1.5,
               label=f'Average ({avg:.1f}/month)')
    best_row = monthly.iloc[best_pos]
    ax.annotate(f'Best: {best_row["count"]}',
                xy=(best_row['month_dt'], best_row['count']),
                xytext=(0, 8), textcoords='offset points',
                ha='center', color=GREEN, fontweight='bold')
    ax.set_title('Workouts per Month', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Sessions logged')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()

    summary = {
        'avg_per_month': round(avg, 1),
        'best_month': str(monthly.iloc[best_pos]['month']),
        'best_count': int(monthly.iloc[best_pos]['count']),
        'worst_month': str(monthly.iloc[worst_pos]['month']),
        'worst_count': int(monthly.iloc[worst_pos]['count']),
    }
    return fig, summary


def weekly_trend(df):
    """Weekly sessions with 4-week rolling average. Returns fig."""
    weekly = df.groupby('week').size().reset_index(name='count')
    weekly['week_dt'] = weekly['week'].dt.to_timestamp()
    weekly['rolling'] = weekly['count'].rolling(4, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(weekly['week_dt'], weekly['count'], alpha=0.2, color=PURPLE)
    ax.plot(weekly['week_dt'], weekly['count'], color=PURPLE, linewidth=1.2)
    ax.plot(weekly['week_dt'], weekly['rolling'], color=AMBER,
            linewidth=2.5, label='4-week rolling avg')
    ax.set_title('Sessions per Week', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Sessions')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    return fig


def day_of_week(df):
    """Day-of-week frequency and RX rate. Returns fig."""
    day_order  = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_counts = df['weekday'].value_counts().reindex(day_order, fill_value=0)
    rx_by_day  = df.groupby('weekday').apply(
        lambda x: (x['rx_or_scaled'] == 'RX').mean() * 100
    ).reindex(day_order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    colors = [GREEN if v == day_counts.max() else BLUE for v in day_counts.values]
    ax1.bar(day_counts.index, day_counts.values, color=colors, alpha=0.85)
    ax1.set_title('Favourite Training Days', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total sessions')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax2.bar(rx_by_day.index, rx_by_day.values, color=AMBER, alpha=0.85)
    ax2.set_title('RX Rate by Day (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('% of sessions RX')
    ax2.set_ylim(0, 100)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    return fig


def detect_gaps(df, threshold_days=7):
    """Return list of (gap_start_date, gap_length_days) tuples."""
    unique_days = (df['date'].dt.normalize()
                   .drop_duplicates().sort_values().reset_index(drop=True))
    diffs = unique_days.diff().dt.days
    gaps  = []
    for idx in diffs[diffs > threshold_days].index:
        gaps.append({
            'start': unique_days.iloc[idx - 1].date(),
            'days':  int(diffs.iloc[idx])
        })
    return gaps


def summary(df):
    """Return dict of key attendance stats for agent use."""
    _, monthly_stats = monthly_chart(df)
    plt.close('all')
    gaps = detect_gaps(df)
    return {
        **monthly_stats,
        'total_sessions': len(df),
        'gaps_over_7d': gaps,
        'date_range': f"{df['date'].min().date()} → {df['date'].max().date()}",
    }
