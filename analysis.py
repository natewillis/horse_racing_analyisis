from models import Races, BettingResults, Entries, EntryPools, Picks, AnalysisProbabilities
from db_utils import get_db_session, shutdown_session_and_engine, load_item_into_database
from arima import run_monte_carlo_arima_on_race
from random_forest import run_monte_carlo_random_forest_on_race
import argparse
import datetime
from joblib import load

def time_frame_definition():

    # Any Useful Variables
    utc_now = datetime.datetime.utcnow()

    # Define dictionary
    date_time_dict = {
        'all time': lambda date: True,
        'last week': lambda date: date > (utc_now - datetime.timedelta(days=7)),
        'all weekdays': lambda date: date.isoweekday() in range(1, 6),
        'all weekends': lambda date: date.isoweekday() in (0, 6)
    }

    return date_time_dict


def write_betting_results_dict_to_database(session, results_dict):
    # Commit Results
    for strategy, tracks in results_dict.items():
        for track, time_frames in tracks.items():
            for time_frame, bet_types in time_frames.items():
                for bet_type, single_result_dict in bet_types.items():

                    # Calculate roi
                    if single_result_dict['bet_cost'] > 0:
                        bet_roi = ((single_result_dict['bet_return'] / single_result_dict['bet_cost']) - 1.0) * 100.0
                    else:
                        bet_roi = 0

                    # Create item
                    item = dict()
                    item['strategy'] = strategy
                    item['bet_type_text'] = bet_type
                    item['time_frame_text'] = time_frame
                    item['track_id'] = track
                    item['bet_count'] = single_result_dict['bet_count']
                    item['bet_cost'] = single_result_dict['bet_cost']
                    item['bet_return'] = single_result_dict['bet_return']
                    item['bet_roi'] = bet_roi
                    item['bet_success_count'] = single_result_dict['bet_success_count']
                    item['update_time'] = single_result_dict['update_time']

                    # Create or update
                    betting_result = load_item_into_database(item, 'betting_result', session)


def straight_favorite_betting(session):
    # Setup variables
    time_now = datetime.datetime.utcnow()
    strategies = ["Straight Favorite Betting"]
    bet_types = [
        'win',
        'place',
        'show',
        'wps'
    ]
    time_frames = time_frame_definition()
    track_list = [track_id[0] for track_id in session.query(Races.track_id).distinct()]
    track_list.append('all')

    # Initialize Results Dictionary
    results_dict = dict()
    for strategy in strategies:
        results_dict[strategy] = dict()
        for track in track_list:
            results_dict[strategy][track] = dict()
            for time_frame, time_frame_function in time_frames.items():
                results_dict[strategy][track][time_frame] = dict()
                for bet_type in bet_types:
                    results_dict[strategy][track][time_frame][bet_type] = dict()
                    results_dict[strategy][track][time_frame][bet_type]['bet_count'] = 0
                    results_dict[strategy][track][time_frame][bet_type]['bet_cost'] = 0
                    results_dict[strategy][track][time_frame][bet_type]['bet_return'] = 0
                    results_dict[strategy][track][time_frame][bet_type]['bet_success_count'] = 0
                    results_dict[strategy][track][time_frame][bet_type]['update_time'] = time_now

    # Get all races
    races = session.query(Races).filter_by(results=True).all()

    # Start looping
    for race in races:

        # Setup Variables
        favorite_entry = None
        win_pool = 0

        # Query Entries
        entries = session.query(Entries). \
            filter_by(race_id=race.race_id). \
            filter_by(scratch_indicator='N').all()

        # Figure out who the favorite is
        for entry in entries:

            # Get win entry pool
            win_entry_pool = session.query(EntryPools). \
                filter_by(entry_id=entry.entry_id). \
                order_by(EntryPools.scrape_time.desc()).first()

            # Determine the favorite
            if win_entry_pool is not None:
                if favorite_entry is None:
                    favorite_entry = entry
                    win_pool = win_entry_pool.amount
                else:
                    if win_entry_pool.amount > win_pool:
                        win_pool = win_entry_pool.amount
                        favorite_entry = entry
            else:
                print(f'Why is the win entry pool None for {entry.entry_id}?')

        # Process Winners
        if favorite_entry is not None:

            # Strategy Loop
            for strategy in strategies:

                for track in ['all', race.track_id]:

                    # Loop through time frames
                    for time_frame, time_frame_function in time_frames.items():

                        # Time Logic
                        if not time_frame_function(race.post_time):
                            continue

                        # Loop through bet types
                        for bet_type in bet_types:

                            if bet_type == 'win':
                                win_amount = favorite_entry.win_payoff
                                bet_cost = 2
                            elif bet_type == 'place':
                                win_amount = favorite_entry.place_payoff
                                bet_cost = 2
                            elif bet_type == 'show':
                                win_amount = favorite_entry.show_payoff
                                bet_cost = 2
                            elif bet_type == 'wps':
                                win_amount = favorite_entry.win_payoff + \
                                             favorite_entry.place_payoff + \
                                             favorite_entry.show_payoff
                                bet_cost = 6

                            # Store Results
                            results_dict[strategy][track][time_frame][bet_type]['bet_count'] += 1
                            results_dict[strategy][track][time_frame][bet_type]['bet_cost'] += bet_cost
                            results_dict[strategy][track][time_frame][bet_type]['bet_return'] += win_amount
                            if win_amount > 0:
                                results_dict[strategy][track][time_frame][bet_type]['bet_success_count'] += 1

    # Write the dictionary to the database
    write_betting_results_dict_to_database(session, results_dict)


def whole_window_favorite_betting(session):
    # Setup variables
    time_now = datetime.datetime.utcnow()
    strategies = [
        "Whole Betting Window Favorite Betting",
        "95% Betting Window Favorite Betting",
        "90% Betting Window Favorite Betting"]
    bet_types = [
        'win',
        'place',
        'show',
        'wps'
    ]
    time_frames = [
        'last 24 hours',
        'last week',
        'all time'
    ]
    results_dict = dict()
    for strategy in strategies:
        results_dict[strategy] = dict()
        for time_frame in time_frames:
            results_dict[strategy][time_frame] = dict()
            for bet_type in bet_types:
                results_dict[strategy][time_frame][bet_type] = dict()
                results_dict[strategy][time_frame][bet_type]['bet_count'] = 0
                results_dict[strategy][time_frame][bet_type]['bet_cost'] = 0
                results_dict[strategy][time_frame][bet_type]['bet_return'] = 0
                results_dict[strategy][time_frame][bet_type]['bet_success_count'] = 0
                results_dict[strategy][time_frame][bet_type]['update_time'] = time_now

    # Get all races
    races = session.query(Races).filter_by(results=True).all()

    # Start looping
    for race in races:

        # Get Entries in race
        entry_ids = [
            entry_id[0] for entry_id in session.query(Entries.entry_id).filter_by(race_id=race.race_id).distinct()
        ]
        entries = session.query(Entries).filter_by(race_id=race.race_id).filter_by(scratch_indicator='N').all()

        # Get unique list of scrape times
        scrape_times = [
            scrape_time[0] for scrape_time in session.query(EntryPools.scrape_time).filter(
                EntryPools.entry_id.in_(entry_ids)
            ).distinct()
        ]

        # Setup Loop Variables
        window_favorites = {}

        # Loop through scrape times to determine favorite
        for scrape_time in scrape_times:

            # Setup Variables
            favorite_entry = None
            win_pool = 0

            # Figure out who the favorite is
            for entry in entries:

                # Get Pool For Entry
                win_entry_pool = session.query(EntryPools). \
                    filter_by(entry_id=entry.entry_id). \
                    filter_by(scrape_time=scrape_time).first()

                # Determine the favorite
                if win_entry_pool is not None:
                    if favorite_entry is None:
                        favorite_entry = entry
                        win_pool = win_entry_pool.amount
                    else:
                        if win_entry_pool.amount > win_pool:
                            win_pool = win_entry_pool.amount
                            favorite_entry = entry
                else:
                    print(f'Why is the win entry pool None for {entry.entry_id}?')

            # Store Favorite for this timestep
            if favorite_entry is not None:
                if str(favorite_entry.entry_id) not in window_favorites:
                    window_favorites[str(favorite_entry.entry_id)] = {
                        'entry': favorite_entry,
                        'count': 0
                    }
                window_favorites[str(favorite_entry.entry_id)]['count'] += 1

        # Check if theres only one entry
        for strategy in strategies:

            bet_entry = None
            for window_favorite_id, window_favorite in window_favorites.items():
                favorite_percentage = window_favorite['count']/len(scrape_times)
                if strategy == 'Whole Betting Window Favorite Betting' and len(window_favorites) == 1:
                    bet_entry = window_favorite['entry']
                elif strategy == '90% Betting Window Favorite Betting' and favorite_percentage >= 0.9:
                    bet_entry = window_favorite['entry']
                elif strategy == '95% Betting Window Favorite Betting' and favorite_percentage >= 0.95:
                    bet_entry = window_favorite['entry']

            if bet_entry is not None:

                # Bring favorite entry back out
                favorite_entry = bet_entry

                # Whole window favorite betting
                # Loop through time frames
                for time_frame in time_frames:

                    # Time Logic
                    if time_frame == 'last 24 hours':
                        if race.post_time < (time_now - datetime.timedelta(days=1)):
                            continue
                    elif time_frame == 'last week':
                        if race.post_time < (time_now - datetime.timedelta(days=7)):
                            continue

                    # Loop through bet types
                    for bet_type in bet_types:

                        if bet_type == 'win':
                            win_amount = favorite_entry.win_payoff
                            bet_cost = 2
                        elif bet_type == 'place':
                            win_amount = favorite_entry.place_payoff
                            bet_cost = 2
                        elif bet_type == 'show':
                            win_amount = favorite_entry.show_payoff
                            bet_cost = 2
                        elif bet_type == 'wps':
                            win_amount = favorite_entry.win_payoff + \
                                         favorite_entry.place_payoff + \
                                         favorite_entry.show_payoff
                            bet_cost = 6

                        # Store Results
                        results_dict[strategy][time_frame][bet_type]['bet_count'] += 1
                        results_dict[strategy][time_frame][bet_type]['bet_cost'] += bet_cost
                        results_dict[strategy][time_frame][bet_type]['bet_return'] += win_amount
                        if win_amount > 0:
                            results_dict[strategy][time_frame][bet_type]['bet_success_count'] += 1


    # Write the dictionary to the database
    write_betting_results_dict_to_database(session, results_dict)


def create_payoff_dictionary(session, race):

    # Entries in order of finish
    entries = session.query(Entries).filter(
        Entries.race_id == race.race_id
    ).order_by(Entries.finish_position).all()

    # Init dictionary
    payoffs = []

    # Win Processing
    if len(entries) > 0:
        if entries[0].win_payoff > 0:
            win_dict = {
                'bet_type': 'WIN',
                'bet_win_text': str(entries[0].program_number),
                'bet_return': entries[0].win_payoff,
                'bet_cost': 2
            }
            payoffs.append(win_dict)

    # Return Finished Dictionary
    return payoffs


def evaluate_picks(session):

    # Find picks that can be evaluated
    picks = session.query(Picks).join(Races).filter(
        Races.drf_results,
        Picks.bet_return == None
    ).all()

    for pick in picks:

        # Get Race
        race = session.query(Races).filter(Races.race_id == pick.race_id).first()

        # Get Race Payoffs
        payoffs = create_payoff_dictionary(session, race)

        # Check if this pick hits any of the payoffs
        if len(payoffs) > 0:
            pick.bet_return = 0
            for payoff in payoffs:
                if payoff['bet_type'] == pick.bet_type and payoff['bet_win_text'] == pick.bet_win_text:

                    # Figure out what our bet was for
                    cost_fraction = pick.bet_cost/payoff['bet_cost']

                    # Figure out winnings
                    pick.bet_return = round(cost_fraction * payoff['bet_return'], 2)

    # Commit update
    session.commit()


def run_arima(session):

    # Get races with history
    races = session.query(Races).filter(
        Races.equibase_horse_results == True,
        Races.drf_entries == True
    ).order_by(Races.card_date.desc()).all()

    # Loop to figure out if analyis is needed
    for race in races:

        # Check if we have an analysis value
        analysis_probability = session.query(AnalysisProbabilities).join(Entries).filter(
            Entries.race_id == race.race_id,
            AnalysisProbabilities.analysis_type == 'arima_stdev_backup'
        ).first()

        if analysis_probability is None:
            run_monte_carlo_arima_on_race(race, session)


def run_random_forest(session):

    # Load Model
    rf_model = load('forest_model.joblib')

    # Get races with history
    races = session.query(Races).filter(
        Races.equibase_horse_results == True,
        Races.drf_entries == True
    ).order_by(Races.card_date.desc()).all()

    # Loop to figure out if analyis is needed
    for race in races:

        # Check if we have an analysis value
        analysis_probability = session.query(AnalysisProbabilities).join(Entries).filter(
            Entries.race_id == race.race_id,
            AnalysisProbabilities.analysis_type == 'random_forest_4_hist'
        ).first()

        if analysis_probability is None:
            run_monte_carlo_random_forest_on_race(race, session, rf_model)


if __name__ == '__main__':

    # Argument Parsing
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode',
                            help="Mode of operation (odds: store odds files)",
                            type=str,
                            required=True,
                            metavar='MODE'
                            )
    args = arg_parser.parse_args()

    # Setup mode tracker
    modes_run = []

    # Check mode
    if args.mode in ('favorite_betting', 'all'):

        # Mode Tracking
        modes_run.append('favorites')

        # Connect to the database
        db_session = get_db_session()

        # Analysis to perform
        straight_favorite_betting(db_session)
        #whole_window_favorite_betting(db_session)

        # Close everything out
        shutdown_session_and_engine(db_session)

    if args.mode in ('all', 'evaluate'):

        # Mode Tracking
        modes_run.append('evaluate')

        # Connect to the database
        db_session = get_db_session()

        # Evaluate picks
        evaluate_picks(db_session)

        # Close everything out
        shutdown_session_and_engine(db_session)

    if args.mode in ('all', 'run_arima', 'analysis_probabilities'):

        # Mode Tracking
        modes_run.append('run_arima')

        # Connect to the database
        db_session = get_db_session()

        # Evaluate picks
        run_arima(db_session)

        # Close everything out
        shutdown_session_and_engine(db_session)

    if args.mode in ('all', 'run_random_forest', 'analysis_probabilities'):

        # Mode Tracking
        modes_run.append('run_random_forest')

        # Connect to the database
        db_session = get_db_session()

        # Evaluate picks
        run_arima(db_session)

        # Close everything out
        shutdown_session_and_engine(db_session)

    if len(modes_run) == 0:

        print(f'"{args.mode}" is not a valid operational mode!')
        exit(1)

    else:

        print(f'We ran the following modes successfully: {",".join(modes_run)}')
