from db_utils import get_db_session, shutdown_session_and_engine, load_item_into_database
from models import Entries, Races
from probabilities import get_win_probabilities_from_monte_carlo_matrix
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import numpy as np


def get_speed_figure_series_from_entry(entry, session):

    # Get race for race date
    race = session.query(Races).filter(Races.race_id==entry.race_id).first()

    # Get all past performance data
    query = session.query(Entries, Races).filter(
        Entries.race_id == Races.race_id,
        Entries.horse_id == entry.horse_id,
        Races.card_date < race.card_date,
        Entries.equibase_speed_figure < 999,
        Entries.equibase_speed_figure > 0,
        Entries.equibase_speed_figure.isnot(None)
    ).order_by(Races.card_date)

    # Assemble list
    speed_figures = []
    for pp_entry, pp_race in query:
        speed_figures.append(pp_entry.equibase_speed_figure)

    # Assemble pandas series
    fake_speed_series = pd.Series(speed_figures, index=pd.date_range(
                                      '2020-01-01',
                                      periods=len(speed_figures),
                                      freq='D'
                                  ))

    # Return it
    return fake_speed_series


def normal_dist_from_entry_arima(entry, session):

    # Set default params
    p = 1
    d = 1
    q = 1

    # Get Speed Figure Time Series
    speed_series = get_speed_figure_series_from_entry(entry, session)

    # Error Check
    if speed_series is None:
        return None, None
    if speed_series.size == 0:
        return None, None
    if speed_series.size == 1:
        return speed_series.values[0], 20

    try:
        # Perform ARIMA
        model = ARIMA(speed_series, order=(p, d, q))
        model_fit = model.fit(disp=False)

        # Perform predictions
        prediction, delta, conf = model_fit.forecast(steps=1, alpha=.318)
        mu = prediction[0]
        scale = (conf[0][1]-conf[0][0])/2
    except ValueError:

        scale = speed_series.std()
        mu = speed_series.mean()

    except ZeroDivisionError:

        scale = speed_series.std()
        mu = speed_series.mean()

    # Return
    return mu, scale


def run_monte_carlo_arima_on_race(race, session):

    # Parameters
    monte_carlo_count = 100000

    # Get Entries
    analyze_entries = session.query(Entries).filter(
        Entries.race_id == race.race_id,
        Entries.scratch_indicator == 'N'
    ).all()

    # Loop through entries
    full_boat_flag = True
    entry_program_numbers = []
    entry_id_numbers = []
    entry_mus = dict()
    race_results = None
    for analyze_entry in analyze_entries:
        entry_program_numbers.append(analyze_entry.program_number)
        entry_id_numbers.append(analyze_entry.entry_id)
        mu, scale = normal_dist_from_entry_arima(analyze_entry, session)

        # Check for None
        if mu is None:
            full_boat_flag = False
            break

        entry_mus[analyze_entry.program_number] = mu
        race_results_entry = np.random.default_rng().normal(mu, scale, monte_carlo_count)
        if race_results is None:
            race_results = race_results_entry
        else:
            race_results = np.vstack((race_results, race_results_entry))

    # Something didn't return a distribution so we can't analyze this race
    if not full_boat_flag:
        print(f'race {race.race_id} cant be analyzed')
        return

    # Get winner probabilities
    winner_probability_items = get_win_probabilities_from_monte_carlo_matrix(
        race_results,
        entry_id_numbers,
        'arima_stdev_backup'
    )

    # Create or update
    for analysis_probability_item in winner_probability_items:
        analysis_probability = load_item_into_database(analysis_probability_item, 'analysis_probability', session)

