"""
loader.py
Central data loading and cleaning. All analysis modules import from here.
"""
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _parse_amrap(raw):
    """Parse SugarWOD AMRAP decimal encoding R.RRR → (rounds, reps).

    SugarWOD stores AMRAP scores as  rounds + reps/1000.
    Examples: 2.010 → (2, 10),  7.062 → (7, 62),  3.000 → (3, 0).
    """
    try:
        f = float(raw)
        rounds = int(f)
        reps = round((f - rounds) * 1000)
        return rounds, reps
    except (ValueError, TypeError):
        return None, None


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

    # ── Score type-aware parsed columns ──────────────────────────────
    # Load (lbs): best_result_raw is already the numeric weight.
    df['score_load'] = pd.to_numeric(
        df['best_result_raw'].where(df['score_type'] == 'Load'),
        errors='coerce',
    )

    # Reps / Calories / Distance: best_result_raw is the raw count.
    countable = df['score_type'].isin(['Reps', 'Calories', 'Feet', 'Meters'])
    df['score_reps'] = pd.to_numeric(
        df['best_result_raw'].where(countable),
        errors='coerce',
    )

    # Time (seconds): best_result_raw would be total seconds.
    # No Time sessions in current export but column kept for forward-compat.
    df['score_seconds'] = pd.to_numeric(
        df['best_result_raw'].where(df['score_type'] == 'Time'),
        errors='coerce',
    )

    # Rounds + Reps: raw = rounds + reps/1000  (e.g. 7.062 → 7 rounds, 62 reps)
    df['score_amrap_rounds'] = None
    df['score_amrap_reps']   = None
    amrap_mask = df['score_type'] == 'Rounds + Reps'
    if amrap_mask.any():
        parsed = df.loc[amrap_mask, 'best_result_raw'].apply(_parse_amrap)
        df.loc[amrap_mask, 'score_amrap_rounds'] = [x[0] for x in parsed]
        df.loc[amrap_mask, 'score_amrap_reps']   = [x[1] for x in parsed]
    df['score_amrap_rounds'] = pd.to_numeric(df['score_amrap_rounds'], errors='coerce')
    df['score_amrap_reps']   = pd.to_numeric(df['score_amrap_reps'],   errors='coerce')

    return df
