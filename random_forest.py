import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import Entries, Horses, Races
from db_utils import get_db_session, shutdown_session_and_engine
from probabilities import get_win_probabilities_from_monte_carlo_matrix
from db_utils import load_item_into_database
import os


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
    race_dates = []
    for pp_entry, pp_race in query:
        speed_figures.append(pp_entry.equibase_speed_figure)
        race_dates.append(pp_race.card_date)

    # Assemble pandas series
    speed_series = pd.Series(speed_figures, index=race_dates)

    # Return it
    return speed_series


def predicted_outcomes_from_entry_random_forest(entry, session, model):

    # Get Speed Figure Time Series
    speed_series = get_speed_figure_series_from_entry(entry, session)

    # Error Check
    if speed_series is None:
        return None
    if speed_series.size < 4:
        return None

    # Numpy Conversion
    speed_array = speed_series.to_numpy()
    speed_array = speed_array[len(speed_array)-4:].reshape(1, -1)

    # Perform random forest predictions
    predictions = [pred.predict(speed_array)[0] for pred in model.estimators_]

    # Return possible outcomes
    return predictions


def get_random_forest_data(session):

    # Create dataframe
    df = pd.DataFrame(columns=[
        'speed_n_m_4',
        'speed_n_m_3',
        'speed_n_m_2',
        'speed_n_m_1',
        'speed_n'
    ])
    df_max = 0

    # Get Entries
    horses = session.query(Horses).filter(Horses.equibase_horse_detail_scrape_date.isnot(None)).all()
    horse_count = 1

    # Loop through horses
    for horse in horses:

        print(f'{horse_count}/{len(horses)} ({horse.horse_name}) with {df_max} rows so far')
        horse_count += 1

        # Get Entries
        entries = session.query(Entries).join(Races).filter(
            Entries.equibase_speed_figure.isnot(None),
            Entries.equibase_speed_figure < 999,
            Entries.horse_id == horse.horse_id
        ).order_by(Races.card_date).all()

        # Ensure we have enough entries
        if len(entries) < 5:
            continue

        # Assemble sliding dataframe
        for i in range(4, len(entries)):
            df.loc[df_max] = [
                entries[i-4].equibase_speed_figure,
                entries[i-3].equibase_speed_figure,
                entries[i-2].equibase_speed_figure,
                entries[i-1].equibase_speed_figure,
                entries[i].equibase_speed_figure
            ]
            df_max += 1

    df.to_csv('temp.csv')


def train_random_forest(session):

    # Get CSV training data
    if not os.path.exists('temp.csv'):
        get_random_forest_data(session)
    features = pd.read_csv('temp.csv')
    features = features.drop(features.columns[0], axis=1)

    # Remove zeros and NANs
    features = features.replace(0, np.nan)
    features = features.dropna()

    # Labels are the values we want to predict
    labels = np.array(features['speed_n'])

    # Remove the labels from the features
    features = features.drop('speed_n', axis=1)

    # Save feature names for later use
    feature_list = list(features.columns)
    print(f'feature list is {feature_list}')

    # Convert to numpy array
    features = features.to_numpy()
    print(features[0])

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42
    )

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    # The baseline predictions are the historical averages
    baseline_preds = test_features[:, feature_list.index('speed_n_m_1')]
    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2))

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Save the model
    dump(rf, 'forest_model.joblib')

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), '.')

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


def run_monte_carlo_random_forest_on_race(race, session, model):
    # Parameters
    monte_carlo_count = 100000

    # Get Entries
    analyze_entries = session.query(Entries).filter(
        Entries.race_id == race.race_id,
        Entries.scratch_indicator == 'N'
    ).order_by(Entries.program_number).all()

    # Loop through entries
    full_boat_flag = True
    entry_program_numbers = []
    entry_id_numbers = []
    entry_mus = dict()
    race_results = None
    for analyze_entry in analyze_entries:
        entry_program_numbers.append(analyze_entry.program_number)
        entry_id_numbers.append(analyze_entry.entry_id)
        outcomes = predicted_outcomes_from_entry_random_forest(analyze_entry, session, model)

        # Check for None
        if outcomes is None:
            full_boat_flag = False
            break

        # Generate random results
        race_results_entry = np.take(outcomes, np.random.randint(low=0, high=len(outcomes), size=monte_carlo_count))
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
        'random_forest_4_hist'
    )

    # Create or update
    for analysis_probability_item in winner_probability_items:
        analysis_probability = load_item_into_database(analysis_probability_item, 'analysis_probability', session)


if __name__ == '__main__':

    db_session = get_db_session()

    # Get Sample Race To Run
    current_race = db_session.query(Races).filter(Races.race_id == 46557).first()

    rf_model = load('forest_model.joblib')
    run_monte_carlo_random_forest_on_race(current_race, db_session, rf_model)

    shutdown_session_and_engine(db_session)
