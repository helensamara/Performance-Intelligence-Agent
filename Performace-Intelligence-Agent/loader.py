"""
loader.py
Central data loading and cleaning. All analysis modules import from here.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_sugarwod(path=None):
    """
    Load and clean SugarWOD CSV export.
    Returns a cleaned DataFrame ready for all analysis modules.
    """
    if path is None:
        path = os.path.join(DATA_DIR, 'workouts.csv')

    df = pd.read_csv(path)

    # Parse dates (handles mixed formats)
    df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False)
    df = df.sort_values('date').reset_index(drop=True)

    # Fill nulls
    df['notes']        = df['notes'].fillna('')
    df['score_type']   = df['score_type'].fillna('Unknown')
    df['barbell_lift'] = df['barbell_lift'].fillna('')
    df['pr']           = df['pr'].fillna('').str.strip()

    # Derived columns
    df['is_pr']   = df['pr'] == 'PR'
    df['week']    = df['date'].dt.to_period('W')
    df['month']   = df['date'].dt.to_period('M')
    df['weekday'] = df['date'].dt.day_name()

    return df
