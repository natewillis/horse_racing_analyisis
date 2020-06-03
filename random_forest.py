import pandas as pd
from models import Entries, Horses, Races
from db_utils import get_db_session, shutdown_session_and_engine

def train_random_forest(session):

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



def dist_from_entry_random_forest(entry, session):
    pass


def run_monte_carlo_random_forest_on_race(race, session):

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


if __name__ == '__main__':

    db_session = get_db_session()

    train_random_forest(db_session)

    shutdown_session_and_engine(db_session)