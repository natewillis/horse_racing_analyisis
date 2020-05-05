from models import db_connect, Races, BettingResults, Entries, EntryPools, create_drf_live_table
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
import argparse
import datetime


def straight_favorite_betting(session):
    # Setup variables
    time_now = datetime.datetime.utcnow()
    strategy = "Straight Favorite Betting"
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
    for time_frame in time_frames:
        results_dict[time_frame] = dict()
        for bet_type in bet_types:
            results_dict[time_frame][bet_type] = dict()
            results_dict[time_frame][bet_type]['bet_count'] = 0
            results_dict[time_frame][bet_type]['bet_cost'] = 0
            results_dict[time_frame][bet_type]['bet_return'] = 0
            results_dict[time_frame][bet_type]['bet_success_count'] = 0
            results_dict[time_frame][bet_type]['update_time'] = time_now

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

            # Determine the favorite
            if favorite_entry is None:
                favorite_entry = entry
            else:
                win_entry_pool = session.query(EntryPools). \
                    filter_by(entry_id=entry.entry_id). \
                    order_by(EntryPools.scrape_time.desc()).first()
                if win_entry_pool is not None:
                    if win_entry_pool.amount > win_pool:
                        win_pool = win_entry_pool.amount
                        favorite_entry = entry

        # Process Winners
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
                    results_dict[time_frame][bet_type]['bet_count'] += 1
                    results_dict[time_frame][bet_type]['bet_cost'] += bet_cost
                    results_dict[time_frame][bet_type]['bet_return'] += win_amount
                    if win_amount > 0:
                        results_dict[time_frame][bet_type]['bet_success_count'] += 1

    # Commit Results
    for time_frame, bet_types in results_dict.items():
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

        # Close everything out
        db_session.close()
        engine.dispose()

    else:

        print(f'"{args.mode}" is not a valid operational mode!')
        exit(1)
