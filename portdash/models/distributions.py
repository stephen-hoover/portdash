from portdash.models import db


class Distribution(db.Model):
    __tablename__ = 'distributions'

    id = db.Column(db.Integer, primary_key=True)


class DistributionType(db.Model):
    __tablename__ = 'distribution_types'

    id = db.Column(db.Integer, primary_key=True)
