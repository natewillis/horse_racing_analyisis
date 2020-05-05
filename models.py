from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL
from settings import DATABASE
from pytz import timezone

# Setup base
base = declarative_base()


def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(URL(**DATABASE))


def create_drf_live_table(engine, destroy_flag):
    """"""
    if destroy_flag:
        base.metadata.drop_all(bind=engine)
    base.metadata.create_all(engine)


class Races(base):
    """Sqlalchemy Races model"""
    __tablename__ = "races"

    race_id = Column(Integer, primary_key=True)

    # Identifying Info
    track_id = Column('track_id', String)
    race_number = Column('race_number', Integer)
    post_time = Column('post_time', DateTime)  # UTC
    day_evening = Column('day_evening', String)
    country = Column('country', String)

    # Race Info
    distance = Column('distance', Float)  # In furlongs
    purse = Column('purse', Integer, nullable=True)
    age_restriction = Column('age_restriction', String, nullable=True)
    race_restriction = Column('race_restriction', String, nullable=True)
    sex_restriction = Column('sex_restriction', String, nullable=True)
    wager_text = Column('wager_text', String, nullable=True)
    race_surface = Column('race_surface', String)
    race_type = Column('race_type', String, nullable=True)
    breed = Column('breed', String, nullable=True)
    track_condition = Column('track_condition', String, nullable=True)

    # Results
    results = Column('results', Boolean, default=False)

    # Scraping Info
    latest_scrape_time = Column('latest_scrape_time', DateTime)  # UTC

    def live_odds_link(self):
        utc_datetime = self.post_time.replace(tzinfo=timezone('UTC'))
        eastern_datetime = utc_datetime.astimezone(timezone('US/Eastern'))
        return (f'http://www.drf.com/liveOdds/tracksPoolDetails'
                f'/currentRace/{self.race_number}'
                f'/trackId/{self.track_id}' 
                f'/country/{self.country}'
                f'/dayEvening/{self.day_evening}'
                f'/date/{eastern_datetime.strftime("%m-%d-%Y")}')

    def results_link(self):
        utc_datetime = self.post_time.replace(tzinfo=timezone('UTC'))
        eastern_datetime = utc_datetime.astimezone(timezone('US/Eastern'))
        return("https://www.drf.com/results/resultDetails/id/" +
               self.track_id + "/country/" + self.country +
               "/date/" + eastern_datetime.strftime("%m-%d-%Y"))


class Horses(base):
    """Sqlalchemy Races model"""
    __tablename__ = "horses"

    horse_id = Column('horse_id', Integer, primary_key=True)

    # Identifying Info
    horse_name = Column('horse_name', String)


class Entries(base):

    """Sqlalchemy Races model"""
    __tablename__ = "entries"

    entry_id = Column('entry_id', Integer, primary_key=True)
    race_id = Column('race_id', Integer, ForeignKey('races.race_id'))
    horse_id = Column('horse_id', Integer, ForeignKey('horses.horse_id'))
    scratch_indicator = Column('scratch_indicator', String)
    post_position = Column('post_position', Integer, nullable=True)
    program_number = Column('program_number', String, nullable=True)

    # Results
    win_payoff = Column('win_payoff', Float, default=0)
    place_payoff = Column('place_payoff', Float, default=0)
    show_payoff = Column('show_payoff', Float, default=0)
    finish_position = Column('finish_position', Integer, nullable=True)


class EntryPools(base):

    """Sqlalchemy Races model"""
    __tablename__ = "entry_pools"

    entry_pool_id = Column('entry_pool_id', Integer, primary_key=True)
    entry_id = Column('entry_id', ForeignKey('entries.entry_id'))
    scrape_time = Column('scrape_time', DateTime)
    pool_type = Column('pool_type', String)
    amount = Column('amount', Float)
    odds = Column('odds', Float, nullable=True)
    dollar = Column('dollar', Float, nullable=True)


class Payoffs(base):
    """Sqlalchemy Races model"""
    __tablename__ = "payoffs"

    payoff_id = Column('entry_pool_id', Integer, primary_key=True)
    race_id = Column('race_id', Integer, ForeignKey('races.race_id'))
    wager_type = Column('wager_type', String)
    wager_type_name = Column('wager_type_name', String)
    winning_numbers = Column('winning_numbers', String)
    number_of_tickets = Column('number_of_tickets', Integer)
    total_pool = Column('total_pool', Integer)
    payoff_amount = Column('payoff_amount', Float)
    base_amount = Column('base_amount', Float)  # Check the wagertypes section


class Probables(base):
    """Sqlalchemy Races model"""
    __tablename__ = "probables"

    probable_id = Column('entry_pool_id', Integer, primary_key=True)
    race_id = Column('race_id', Integer, ForeignKey('races.race_id'))
    scrape_time = Column('scrape_time', DateTime)
    probable_type = Column('probable_type', String)
    program_numbers = Column('program_numbers', String)
    probable_value = Column('probable_value', Float)
    probable_pool_amount = Column('probable_pool_amount', Float)


class BettingResults(base):
    """Sqlalchemy Races model"""
    __tablename__ = "betting_results"

    betting_result_id = Column('betting_result_id', Integer, primary_key=True)
    strategy = Column('strategy', String)
    bet_type_text = Column('bet_type_text', String)
    time_frame_text = Column('time_frame_text', String)
    bet_count = Column('bet_count', Integer)
    bet_cost = Column('bet_cost', Float)
    bet_return = Column('bet_return', Float)
    bet_roi = Column('bet_roi', Float)
    bet_success_count = Column('bet_success_count', Integer)
    update_time = Column('update_time', DateTime)
