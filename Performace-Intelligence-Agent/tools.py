"""
tools.py
Agent-facing tools. Each function wraps an analysis module and returns
a JSON-serialisable dict that Claude can reason over.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from loader import load_sugarwod
from analysis import attendance, sentiment, performance, ml_models

_df = None

def _get_df():
    global _df
    if _df is None:
        _df = load_sugarwod()
        _df = sentiment.enrich(_df)
    return _df


def tool_attendance_summary():
    """Attendance stats: sessions/month, gaps, best/worst periods."""
    return attendance.summary(_get_df())


def tool_sentiment_summary():
    """Sentiment stats: avg score, correlation with RX, top positive/negative sessions."""
    return sentiment.summary(_get_df())


def tool_performance_summary():
    """Performance stats: RX rate, PR count, scaling ratio trend."""
    return performance.summary(_get_df())


def tool_cluster_workouts():
    """KMeans cluster profiles — workout archetypes with counts and RX/PR rates."""
    _, profiles, _ = ml_models.cluster_workouts(_get_df())
    plt.close('all')
    return profiles[['name','count','rx_rate','pr_rate']].to_dict('index')


def tool_detect_anomalies():
    """Isolation Forest — list of flagged unusual strength sessions."""
    anom, _ = ml_models.detect_anomalies(_get_df())
    plt.close('all')
    return anom.assign(date=anom['date'].astype(str)).to_dict('records')


def tool_forecast_prs():
    """Linear Regression — forecasted days to next PR per lift."""
    forecasts, _ = ml_models.forecast_prs(_get_df())
    plt.close('all')
    return forecasts


# ── Registry ──────────────────────────────────────────────────────

TOOLS = {
    'attendance_summary':  tool_attendance_summary,
    'sentiment_summary':   tool_sentiment_summary,
    'performance_summary': tool_performance_summary,
    'cluster_workouts':    tool_cluster_workouts,
    'detect_anomalies':    tool_detect_anomalies,
    'forecast_prs':        tool_forecast_prs,
}

TOOL_SCHEMAS = [
    {'name': 'attendance_summary',
     'description': 'Get attendance stats: sessions per month, rest gaps over 7 days, best/worst training periods.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'sentiment_summary',
     'description': 'Analyse sentiment in workout notes. Returns avg score, correlation with RX rate, most positive/negative sessions.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'performance_summary',
     'description': 'Get performance stats: overall RX rate, PR count, scaling ratio (actual vs prescribed weight) trend.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'cluster_workouts',
     'description': 'Use KMeans to group workouts into archetypes: Strength, Timed Metcon, AMRAP, Accessory.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'detect_anomalies',
     'description': 'Use Isolation Forest to flag statistically unusual strength sessions worth investigating.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
    {'name': 'forecast_prs',
     'description': 'Use Linear Regression on lift progression curves to forecast when the next PR is likely per lift.',
     'input_schema': {'type': 'object', 'properties': {}, 'required': []}},
]


def run_tool(name, _input=None):
    if name not in TOOLS:
        return {'error': f'Unknown tool: {name}'}
    return TOOLS[name]()
