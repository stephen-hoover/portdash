import pandas as pd

from portdash.extensions import db


class Quote(db.Model):
    __tablename__ = 'quotes'

    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float, nullable=True)

    security_id = db.Column(db.Integer, db.ForeignKey('securities.id'),
                            nullable=False)
    security = db.relationship('Security',
                               backref=db.backref('quotes', lazy='select'))

    @classmethod
    def get_price(cls, symbol, index=None):
        # TODO: Test this function
        if index is None:
            sql_filter = Quote.security.has(symbol=symbol)
        else:
            sql_filter = db.and_(Quote.security.has(symbol=symbol),
                                 Quote.date >= index.min(),
                                 Quote.date <= index.max())
        df = pd.read_sql(sql=(db.session
                              .query(cls)
                              .filter(sql_filter)
                              .with_entities(cls.date, cls.price)
                              .statement),
                         con=db.engine,
                         parse_dates=['date'],
                         index_col='date')
        if index is not None:
            # Use "ffill" to take us through e.g. weekends. Use bfill to
            # make sure that we have valid prices at the beginning of the series.
            price = (df['price']
                     .reindex(index, method='ffill')
                     .fillna(method='ffill')
                     .fillna(method='bfill'))
        else:
            price = df['price']
        return price

    def __repr__(self):
        return (f'Quote(date={self.date}, price={self.price}, '
                f'volume={self.volume}, security_id={self.security_id})')

    def __str__(self):
        return f'{self.security.symbol} is {self.price} on {self.date}'
