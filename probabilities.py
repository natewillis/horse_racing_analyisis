import numpy as np
from models import Entries, Races, AnalysisProbabilities

from db_utils import load_item_into_database
import os
from joblib import load


def generate_probabilities(session, model_dict_list):

    # Get base directory for model files
    models_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    for model_dict in model_dict_list:

        # Load it!
        if model_dict['needs_loading']:
            model_file_path = os.path.join(models_base, model_dict['model_name'] + '.joblib')
            if os.path.exists(model_file_path):
                model_dict['loaded_model'] = load(filename=model_file_path)
            else:
                print(f'{model_dict["model_name"]} doesnt exist')
                model_dict['training_function'](session, model_dict)
                model_dict['loaded_model'] = load(filename=model_file_path)

        # Get races with history
        races = session.query(Races).filter(
            Races.drf_entries == True
        ).order_by(Races.card_date.desc()).all()

        # Loop to figure out if analysis is needed
        number_of_races = len(races)
        for race_number, race in enumerate(races, 1):

            # Status
            print(f'Processing race {race_number}/{number_of_races} ({race.race_id})')

            # Check if we have an analysis value
            analysis_probability = session.query(AnalysisProbabilities).join(Entries).filter(
                Entries.race_id == race.race_id,
                AnalysisProbabilities.analysis_type == model_dict['model_name']
            ).first()

            if analysis_probability is None:
                generate_win_probabilities_from_stored_model(race, session, model_dict)

        # Unload model
        if 'loaded_model' in model_dict:
            del model_dict['loaded_model']


def generate_win_probabilities_from_stored_model(race, session, model_dict):

    # Get Entries
    analyze_entries = session.query(Entries).filter(
        Entries.race_id == race.race_id,
        Entries.scratch_indicator == 'N'
    ).all()

    # Loop variables
    entry_id_numbers = []
    speed_figure_matrix = None

    # Loop through entries
    for analyze_entry in analyze_entries:

        # Collect id numbers
        entry_id_numbers.append(analyze_entry.entry_id)

        # Generate speed figures
        monte_carlo_speed_figures = model_dict['monte_carlo_speed_figure_function'](analyze_entry, session, model_dict)

        # Error check
        if monte_carlo_speed_figures is None:
            print(f'The model was unable to create speed figures in race {race.race_id} for entry {analyze_entry.entry_id}')
            return None

        # Append them to the matrix
        if speed_figure_matrix is None:
            speed_figure_matrix = monte_carlo_speed_figures
        else:
            speed_figure_matrix = np.vstack((speed_figure_matrix, monte_carlo_speed_figures))

    # generate win probabilities from monte carlo speed figure matrix
    winner_probability_items = get_win_probabilities_from_monte_carlo_speed_figure_matrix(
        speed_figure_matrix,
        entry_id_numbers,
        model_dict['model_name']
    )

    # Create or update items in database
    for analysis_probability_item in winner_probability_items:
        analysis_probability = load_item_into_database(analysis_probability_item, 'analysis_probability', session)


def get_win_probabilities_from_monte_carlo_speed_figure_matrix(race_result_matrix, entry_id_list, analysis_type):

    # Setup Return Variables
    return_list = []

    # Matrix Size
    entry_count, race_count = race_result_matrix.shape

    # Winner counting
    winners = np.argmax(race_result_matrix, axis=0)
    indexes, counts = np.unique(winners, return_counts=True)

    # Loop through results
    for index, count in zip(indexes, counts):

        # Division
        if count:
            probability_percentage = float(count) / float(race_count)
        else:
            probability_percentage = 0

        # Create Item
        item = {
            'entry_id': entry_id_list[index],
            'finish_place': 1,
            'probability_percentage': probability_percentage,
            'analysis_type': analysis_type
        }

        # Append to list
        return_list.append(item)

    if len(return_list) != entry_count:
        for entry_id in entry_id_list:
            if next((item for item in return_list if item["entry_id"] == entry_id), None) is None:
                return_list.append({
                    'entry_id': entry_id,
                    'finish_place': 1,
                    'probability_percentage': 0,
                    'analysis_type': analysis_type
                })

    # Return
    return return_list
