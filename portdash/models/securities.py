import enum

from portdash.extensions import db


class LocationEnum(enum.Enum):
    domestic = 'Domestic'
    international = 'International'
    emerging_markets = 'Emerging Markets'


class Security(db.Model):
    __tablename__ = 'securities'

    symbol = db.Column(db.String(10), primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    location = db.Column(db.Enum(LocationEnum), nullable=True)

    # Where can we find quotes for this security?
    source = db.Column(db.String(100), nullable=False)

    # The last time we updated the security's quotes in the database.
    last_updated = db.Column(db.DateTime, nullable=True)

    def __repr__(self):
        return (f'Security(symbol="{self.symbol}", name="{self.name}", '
                f'type="{self.type}", location="{self.location}", '
                f'source="{self.source}", last_updated={self.last_updated})')
