from models import db_connect, Races, BettingResults, Entries, EntryPools, create_drf_live_table
from sqlalchemy.orm import sessionmaker
import argparse
import datetime


def write_betting_results_dict_to_database(session, results_dict):
    # Commit Results
    for strategy, time_frames in results_dict.items():
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
                item['bet_count'] = single_result_dict['bet_count']
                item['bet_cost'] = single_result_dict['bet_cost']
                item['bet_return'] = single_result_dict['bet_return']
                item['bet_roi'] = bet_roi
                item['bet_success_count'] = single_result_dict['bet_success_count']
                item['update_time'] = single_result_dict['update_time']

                # Check for existing record
                betting_result = session.query(BettingResults).filter(
                    BettingResults.time_frame_text == item['time_frame_text'],
                    BettingResults.bet_type_text == item['bet_type_text'],
                    BettingResults.strategy == item['strategy']
                ).first()

                # New item logic
                if betting_result is None:

                    # Process new item
                    betting_result = BettingResults(**item)

                # Otherwise update existing item
                else:

                    # Set the new attributes
                    for key, value in item.items():
                        setattr(betting_result, key, value)

                # Add to session
                session.add(betting_result)

    # Commit session
    session.commit()


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
        for strategy in strategies:
            if favorite_entry is not None:

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

    # Check mode
    if args.mode in ('all'):

        # Connect to the database
        engine = db_connect()
        create_drf_live_table(engine, False)
        session_maker_class = sessionmaker(bind=engine)
        db_session = session_maker_class()

        # Analysis to perform
        straight_favorite_betting(db_session)
        whole_window_favorite_betting(db_session)

        # Close everything out
        db_session.close()
        engine.dispose()

    else:

        print(f'"{args.mode}" is not a valid operational mode!')
        exit(1)
