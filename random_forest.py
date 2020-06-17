import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import Entries, Horses, Races, Workouts
from db_utils import get_db_session, shutdown_session_and_engine
from probabilities import get_win_probabilities_from_monte_carlo_speed_figure_matrix
from db_utils import load_item_into_database
import os
import datetime


def get_speed_figure_random_forest_data_4_last_speed_figs(session):

    # Specs
    data_name = 'random_forest_data_4_last_speed_figs'

    # Get base directory for model files
    models_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

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
    horses = session.query(Horses).filter(
        Horses.equibase_horse_detail_scrape_date.isnot(None),
        Horses.horse_birthday.isnot(None)
    ).all()
    horse_count = 1

    # Loop through horses
    for horse in horses:

        print(f'{horse_count}/{len(horses)} ({horse.horse_name}) with {df_max} rows so far')
        horse_count += 1

        # Get Entries
        entry_query = session.query(Entries, Races).filter(
            Entries.race_id == Races.race_id,
            Entries.equibase_speed_figure.isnot(None),
            Entries.equibase_speed_figure < 999,
            Entries.horse_id == horse.horse_id,
            Races.card_date < (datetime.date.today() - datetime.timedelta(days=7))
        ).order_by(Races.card_date).all()

        # Unpack the query in a way we can deal with it
        entries = []
        races = []
        for entry, race in entry_query:
            entries.append(entry)
            races.append(race)

        # Ensure we have enough entries
        if len(entries) < 5:
            print(f'not processing any of the {len(entries)} past performances for {horse.horse_name}')
            continue

        # Assemble sliding dataframe
        for i in range(4, len(entries)):
            df.loc[df_max] = [
                entries[i - 4].equibase_speed_figure,
                entries[i - 3].equibase_speed_figure,
                entries[i - 2].equibase_speed_figure,
                entries[i - 1].equibase_speed_figure,
                entries[i].equibase_speed_figure
            ]
            df_max += 1

    # Write saved data file
    df.to_csv(os.path.join(models_base, f'{data_name}.csv'))


def get_speed_figure_random_forest_data_3_last_speed_figs_days_since_race_horse_age(session):

    # Specs
    data_name = 'random_forest_data_3_last_speed_figs_days_since_race_horse_age'

    # Get base directory for model files
    models_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    # Create dataframe
    df = pd.DataFrame(columns=[
        'horse_age',
        'days_to_race_n_m_3',
        'speed_n_m_3',
        'days_to_race_n_m_2',
        'speed_n_m_2',
        'days_to_race_n_m_1',
        'speed_n_m_1',
        'speed_n'
    ])
    df_max = 0

    # Get Entries
    horses = session.query(Horses).filter(
        Horses.equibase_horse_detail_scrape_date.isnot(None),
        Horses.horse_birthday.isnot(None)
    ).all()
    horse_count = 1

    # Loop through horses
    for horse in horses:

        print(f'{horse_count}/{len(horses)} ({horse.horse_name}) with {df_max} rows so far')
        horse_count += 1

        # Get Entries
        entry_query = session.query(Entries, Races).filter(
            Entries.race_id == Races.race_id,
            Entries.equibase_speed_figure.isnot(None),
            Entries.equibase_speed_figure < 999,
            Entries.horse_id == horse.horse_id,
            Races.card_date < (datetime.date.today() - datetime.timedelta(days=7))
        ).order_by(Races.card_date).all()

        # Unpack the query in a way we can deal with it
        entries = []
        races = []
        for entry, race in entry_query:
            entries.append(entry)
            races.append(race)

        # Ensure we have enough entries
        if len(entries) < 4:
            print(f'not processing any of the {len(entries)} past performances for {horse.horse_name}')
            continue

        # Assemble sliding dataframe
        for i in range(3, len(entries)):
            df.loc[df_max] = [
                (races[i].card_date - horse.horse_birthday).days,
                (races[i].card_date - races[i-3].card_date).days,
                entries[i-3].equibase_speed_figure,
                (races[i].card_date - races[i - 2].card_date).days,
                entries[i-2].equibase_speed_figure,
                (races[i].card_date - races[i - 1].card_date).days,
                entries[i-1].equibase_speed_figure,
                entries[i].equibase_speed_figure
            ]
            df_max += 1

    # Write saved data file
    df.to_csv(os.path.join(models_base, f'{data_name}.csv'))


def train_random_forest_last_four_speed_figures(session, model_dict):

    training_config = {
        'n_estimators': 100,
        'max_depth': 4,
        'data_file_name': 'random_forest_data_4_last_speed_figs.csv',
        'data_file_function': get_speed_figure_random_forest_data_4_last_speed_figs
    }
    train_speed_figure_random_forest(session, model_dict, training_config)


def train_random_forest_relative_last_four_speed_figures(session, model_dict):

    training_config = {
        'n_estimators': 400,
        'max_depth': 10,
        'data_file_name': 'random_forest_data_4_last_speed_figs.csv',
        'data_file_function': get_speed_figure_random_forest_data_4_last_speed_figs
    }
    train_speed_figure_random_forest(session, model_dict, training_config)


def train_random_forest_last_four_relative_speed_figures(session, model_dict):

    training_config = {
        'n_estimators': 400,
        'max_depth': 10,
        'data_file_name': 'random_forest_data_4_last_speed_figs.csv',
        'data_file_function': get_speed_figure_random_forest_data_4_last_speed_figs,
        'post_processing_function': create_relative_speed_ratings
    }
    train_speed_figure_random_forest(session, model_dict, training_config)


def train_random_forest_last_three_speed_figs_and_days_back_with_horse_age(session, model_dict):
    training_config = {
        'n_estimators': 300,
        'max_depth': 7,
        'data_file_name': 'random_forest_data_3_last_speed_figs_days_since_race_horse_age.csv',
        'data_file_function': get_speed_figure_random_forest_data_3_last_speed_figs_days_since_race_horse_age
    }
    train_speed_figure_random_forest(session, model_dict, training_config)


def create_relative_speed_ratings(speed_df):

    # Create column list
    speed_column_list = list(speed_df.columns)
    relative_column_list = []

    # Create copy of original dataframe
    relative_df = speed_df.copy()

    # Loop through columns
    for column_1, column_2 in zip(speed_column_list[:-1], speed_column_list[1:]):

        # Create relative columns
        relative_column_name = column_2 + '_relative'
        relative_column_list.append(relative_column_name)
        relative_df[relative_column_name] = relative_df[column_2]-relative_df[column_1]

    # Delete old columns
    relative_df = relative_df.drop(speed_column_list[0], axis=1)
    for relative_column_name, speed_column_name in zip(relative_column_list, speed_column_list[1:]):
        relative_df = relative_df.drop(speed_column_name, axis=1)
        relative_df = relative_df.rename(columns={relative_column_name: speed_column_name})

    # Return processed data
    return relative_df


def turn_relative_speed_figures_to_speed_figures(relative_speed_figures, entry, session):

    speed_figure_series = get_whole_model_input_series_from_entry_random_forest_v1(entry, session)

    return speed_figure_series[-1] + relative_speed_figures

    pass


def train_speed_figure_random_forest(session, model_dict, training_config):

    # Get base directory for model files
    models_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    data_file = os.path.join(models_base, training_config['data_file_name'])

    # Get CSV training data
    if not os.path.exists(data_file):
        training_config['data_file_function'](session)
    features = pd.read_csv(data_file)
    features = features.drop(features.columns[0], axis=1)

    # Remove zeros and NANs
    features = features.replace(0, np.nan)
    features = features.dropna()

    # Post processing if necessary
    if 'post_processing_function' in training_config:
        features = training_config['post_processing_function'](features)

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
    baseline_preds = np.mean(test_features[:, [feature_list.index('speed_n_m_1'), feature_list.index('speed_n_m_2'), feature_list.index('speed_n_m_3')]])
    baseline_errors = baseline_preds - test_labels
    print('Average baseline error: ', np.mean(baseline_errors))

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=training_config['n_estimators'], max_depth=training_config['max_depth'])

    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Save the model
    dump(rf, os.path.join(models_base, model_dict['model_name'] + '.joblib'))

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Calculate the absolute errors
    errors = predictions - test_labels

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', np.mean(errors), '.')

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
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


def get_whole_model_input_series_from_entry_random_forest_v1(entry, session):

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


def model_input_random_forest_last_four_speed_figures(entry, session):

    # Get Speed Figure Time Series
    speed_series = get_whole_model_input_series_from_entry_random_forest_v1(entry, session)

    # Error Check
    if speed_series is None:
        print(f'speed series was none for entry {entry.entry_id}')
        return None
    if speed_series.size < 4:
        print(f'speed series was size {speed_series.size} for entry {entry.entry_id}')
        return None

    # Numpy Conversion
    speed_array = speed_series.to_numpy()
    speed_array = speed_array[len(speed_array) - 4:].reshape(1, -1)

    # Return model inputs
    return speed_array


def model_input_random_forest_last_four_relative_speed_figures(entry, session):

    # Get Speed Figure Time Series
    speed_series = get_whole_model_input_series_from_entry_random_forest_v1(entry, session)

    # Error Check
    if speed_series is None:
        print(f'speed series was none for entry {entry.entry_id}')
        return None
    if speed_series.size < 4:
        print(f'speed series was size {speed_series.size} for entry {entry.entry_id}')
        return None

    # Numpy Conversion
    speed_array = speed_series.to_numpy()
    speed_array = speed_array[len(speed_array) - 4:].reshape(1, -1)
    speed_array = (speed_array[0][1:] - speed_array[0][:-1]).reshape(1, -1)

    # Return model inputs
    return speed_array


def generate_single_entry_monte_carlo_speed_figures_from_random_forest_last_four_speed_figures(entry, session, model_dict):

    # Wrapper for main random forest function with input function specified
    return generate_single_entry_monte_carlo_speed_figures_from_random_forest(
        entry,
        session,
        model_dict,
        model_input_random_forest_last_four_speed_figures
    )


def generate_single_entry_monte_carlo_speed_figures_from_random_forest_last_four_relative_speed_figures(entry, session, model_dict):

    # Wrapper for main random forest function with input function specified
    return generate_single_entry_monte_carlo_speed_figures_from_random_forest(
        entry,
        session,
        model_dict,
        model_input_random_forest_last_four_relative_speed_figures
    )


def generate_single_entry_monte_carlo_speed_figures_from_random_forest(entry, session, model_dict, model_input_function):

    # Error checking
    if model_dict['loaded_model'] is None:
        return None

    # Generate random forest model inputs
    model_input = model_input_function(entry, session)
    if model_input is None:
        return None

    # Perform random forest predictions
    predicted_outcomes = [current_tree.predict(model_input)[0] for current_tree in model_dict['loaded_model'].estimators_]

    # Check for None
    if predicted_outcomes is None:
        return None

    # Generate random results
    speed_figures = np.take(
        predicted_outcomes,
        np.random.randint(low=0, high=len(predicted_outcomes), size=model_dict['number_of_speed_figures'])
    )

    if 'speed_figure_model_input_post_processing_function' in model_dict:
        speed_figures = model_dict['speed_figure_model_input_post_processing_function'](speed_figures, entry, session)

    # Return speed figures
    return speed_figures


if __name__ == '__main__':

    db_session = get_db_session()

    train_speed_figure_random_forest(db_session)

    shutdown_session_and_engine(db_session)
