import numpy as np


def get_win_probabilities_from_monte_carlo_matrix(race_result_matrix, entry_id_list, analysis_type):

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
