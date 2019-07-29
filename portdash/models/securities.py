from portdash.extensions import db


class Security(db.Model):
    __tablename__ = 'securities'

    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(150), nullable=True)

    def __repr__(self):
        return f'Security(symbol="{self.symbol}", name="{self.name}")'
