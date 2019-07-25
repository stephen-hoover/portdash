from portdash.models import db


class Quote(db.Model):
    __tablename__ = 'quotes'

    id = db.Column(db.Integer, primary_key=True)
    security_id = db.Column(db.Integer)
    price = db.Column(db.Float)
