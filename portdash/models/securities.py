from portdash.extensions import db


class Security(db.Model):
    __tablename__ = 'securities'

    symbol = db.Column(db.String(10), primary_key=True)
    name = db.Column(db.String(150), nullable=True)

    def __repr__(self):
        return f'Security(symbol="{self.symbol}", name="{self.name}")'
