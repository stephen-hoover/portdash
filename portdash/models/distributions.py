from portdash.extensions import db


class Distribution(db.Model):
    __tablename__ = 'distributions'

    id = db.Column(db.Integer, primary_key=True)
    type_id = db.Column(db.Integer)
    security_id = db.Column(db.Integer)
    amount = db.Column(db.Float)


class DistributionType(db.Model):
    __tablename__ = 'distribution_types'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
