import logging
from typing import Sequence, Union

import pandas as pd

from portdash.extensions import db

log = logging.getLogger(__name__)


class Quote(db.Model):
    __tablename__ = 'quotes'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=True)
    # NB: Update `insert_prices` when adding new data columns

    symbol = db.Column(db.Integer, db.ForeignKey('securities.symbol'),
                       nullable=False)
    security = db.relationship('Security',
                               backref=db.backref('quotes', lazy='select'))

    @classmethod
    def insert(cls, df: pd.DataFrame, symbol: str):
        """Add new prices to the quotes table without creating duplicates

        This function checks for existing prices and only adds rows
        for dates which do not yet exist for the given security.

        Parameters
        ----------
        df : pd.DataFrame
            Table indexed by date, with columns "price" and "volume".
        symbol : str
            The ticker symbol for the security to update.
        """
        if 'price' not in df or 'volume' not in df:
            raise TypeError(f'The input table must have "price" and '
                            f'"volume" columns.')
        sql = (db.session.query(cls)
               .filter(cls.symbol == symbol)
               .with_entities(cls.date, cls.price)
               .statement)
        existing = pd.read_sql(sql=sql, con=db.engine, parse_dates=['date'],
                               index_col='date')
        to_insert = df.loc[df.index.difference(existing.index)]
        if not len(to_insert):
            log.debug(f'No new rows to insert for {symbol}.')
            return

        (to_insert[['price', 'volume']]
         .assign(symbol=symbol)
         .to_sql(name=cls.__tablename__, con=db.engine, index=True,
                 index_label='date', if_exists='append', method='multi',
                 chunksize=100))
        log.debug(f"Inserted {len(to_insert)} rows of prices for {symbol}.")

    @classmethod
    def most_recent(cls, symbol: str=None) -> 'Quote':
        if symbol:
            qu = (cls.query
                  .filter(cls.symbol == symbol)
                  .order_by(cls.date.desc()).first())
        else:
            qu = (cls.query
                  .order_by(cls.date.desc()).first())
        return qu

    @classmethod
    def get_price_at_date(cls, symbol: str, date: pd.datetime) -> float:
        row = (cls.query
               .filter(db.and_(cls.symbol == symbol, cls.date == date))
               .first())
        if not row:
            log.debug(f"Security {symbol} has no price for {date}.")
            return
        else:
            return row.price

    @classmethod
    def get_price(cls,
                  symbol: str,
                  index: Union[pd.DatetimeIndex,
                               Sequence[pd.datetime]]=None) -> pd.Series:
        df = pd.read_sql(sql=(db.session
                              .query(cls)
                              .filter(cls.symbol == symbol)
                              .with_entities(cls.date, cls.price)
                              .statement),
                         con=db.engine,
                         parse_dates=['date'],
                         index_col='date')
        if index is not None:
            # Use "ffill" to take us through e.g. weekends. Use bfill to
            # make sure that we have valid prices at
            # the beginning of the series.
            price = (df['price']
                     .reindex(index, method='ffill')
                     .fillna(method='ffill')
                     .fillna(method='bfill'))
        else:
            price = df['price']
        return price

    def __repr__(self):
        return (f'Quote(date={self.date}, price={self.price}, '
                f'volume={self.volume}, symbol={self.symbol})')

    def __str__(self):
        return f'{self.symbol} is {self.price} on {self.date}'
