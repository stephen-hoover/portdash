from datetime import datetime
import logging
from typing import Sequence, Union

import pandas as pd

from portdash.extensions import db

log = logging.getLogger(__name__)


class Distribution(db.Model):
    __tablename__ = 'distributions'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    amount = db.Column(db.Float, nullable=False)

    symbol = db.Column(db.Integer, db.ForeignKey('securities.symbol'),
                       nullable=False)
    security = db.relationship('Security',
                               backref=db.backref('distributions',
                                                  lazy='select'))

    @classmethod
    def insert(cls, df: pd.DataFrame, symbol: str):
        """Add new amounts to the distributions table without duplicates

        This function checks for existing amounts and only adds rows
        for dates which do not yet exist for the given security and
        which have non-zero distribution amounts.

        Parameters
        ----------
        df : pd.DataFrame
            Table indexed by date, with column "amount" or "dividend_amount".
        symbol : str
            The ticker symbol for the security to update.
        """
        if 'amount' not in df and 'dividend_amount' in df:
            df = df.rename(columns={'dividend_amount': 'amount'})
        if 'amount' not in df:
            raise TypeError(f'The input table must have an "amount" or '
                            f'"dividend_amount" column.')
        sql = (db.session.query(cls)
               .filter(cls.symbol == symbol)
               .with_entities(cls.date, cls.amount)
               .statement)
        existing = pd.read_sql(sql=sql, con=db.engine, parse_dates=['date'],
                               index_col='date')
        to_insert = df.loc[df.index.difference(existing.index)]
        if not len(to_insert):
            log.debug(f'No new rows to insert for {symbol}.')
            return
        to_insert = to_insert[to_insert['amount'] != 0]

        (to_insert[['amount']]
         .assign(symbol=symbol)
         .to_sql(name=cls.__tablename__, con=db.engine, index=True,
                 index_label='date', if_exists='append', method='multi',
                 chunksize=100))
        log.debug(f"Inserted {len(to_insert)} distributions for {symbol}.")

    @classmethod
    def most_recent(cls, symbol: str=None) -> 'Distribution':
        if symbol:
            qu = (cls.query
                  .filter(cls.symbol == symbol)
                  .order_by(cls.date.desc()).first())
        else:
            qu = (cls.query
                  .order_by(cls.date.desc()).first())
        return qu

    @classmethod
    def get_amount_at_date(cls, symbol: str, date: datetime) -> float:
        row = (cls.query
               .filter(db.and_(cls.symbol == symbol, cls.date == date))
               .first())
        if not row:
            log.debug(f"Security {symbol} has no distribution for {date}.")
            return
        else:
            return row.amount

    @classmethod
    def get_amount(cls,
                   symbol: str,
                   index: Union[pd.DatetimeIndex,
                               Sequence[datetime]]=None) -> pd.Series:
        df = pd.read_sql(sql=(db.session
                              .query(cls)
                              .filter(cls.symbol == symbol)
                              .with_entities(cls.date, cls.amount)
                              .statement),
                         con=db.engine,
                         parse_dates=['date'],
                         index_col='date')
        if index is not None:
            amount = df.loc[(df.index >= index.min()) &
                            (df.index <= index.max()), 'amount']
        else:
            amount = df['amount']
        return amount

    def __repr__(self):
        return (f'Distribution(date={self.date}, amount={self.amount}, '
                f'symbol={self.symbol})')

    def __str__(self):
        return (f'{self.symbol} distributed {self.amount} '
                f'per share on {self.date}')
