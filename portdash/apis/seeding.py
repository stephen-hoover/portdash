"""For batch loading of data into the database
"""
import logging
from typing import Dict, List

from portdash.io import symbol_lookup, UnknownSymbol
from portdash.io.quotes import DEFAULT_QUOTE_SOURCE
from portdash.extensions import db
from portdash.models import Security

log = logging.getLogger(__name__)


def seed_securities(sec_list: List[Dict[str, str]]):
    """Store a list of dictionaries describing securities in the database


    Parameters
    ----------
    sec_list:
     Each element of the list must be a dictionary having at least the
     "symbol" key. If that security does not exist in the database, it
     will be added. If it does exist, its entries will be updated from
     the input dictionary.
    """
    columns = set(Security.__table__.columns.keys())
    try:
        for sec_desc in sec_list:
            if 'symbol' not in sec_desc:
                log.error(f'No "symbol" in the securities table seed '
                          f'{sec_desc}. Skipping.')
            unknown_keys = set(sec_desc) - columns
            if unknown_keys:
                log.warning(f'Found seed attributes {unknown_keys} which do '
                            f'not correspond to any of the columns in the '
                            f'securities table: {columns}. Ignoring extra '
                            f'attributes.')
            sec = Security.query.get(sec_desc['symbol'])
            if not sec:
                sec = _new_security(sec_desc['symbol'])
            for col_name in (set(sec_desc) & columns):
                setattr(sec, col_name, sec_desc[col_name])
        db.session.commit()
    except Exception:
        log.exception(f'Unable to seed securities. Rolling back transactions.')
        db.session.rollback()


def _new_security(symbol: str) -> Security:
    """Insert a new Security in the database, using type and name data
    retrieved from the web API.
    """
    try:
        sec_data = symbol_lookup(symbol)
    except UnknownSymbol as exc:
        log.warning(f'Could not find a match online for {symbol}. '
                    f'Best matches : {exc.matches}.')
        sec_data = {'name': symbol}
    sec = Security(symbol=symbol, type=sec_data.get('type'),
                   name=sec_data['name'], quote_source=DEFAULT_QUOTE_SOURCE)
    db.session.add(sec)
    return sec


if __name__ == '__main__':
    # When run as a script, seed everything.
    import argparse
    import config
    from portdash import create_app

    parser = argparse.ArgumentParser(
        description="Load data into the database from a config file")
    parser.add_argument('-c', '--conf', required=True,
                        help="Configuration file in YAML format")
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help="Output verbose logs.")
    args = parser.parse_args()

    logging.basicConfig(level=('DEBUG' if args.verbose else 'INFO'))
    config.load_config(args.conf)

    server = create_app()

    with server.app_context():
        db.create_all()
        securities_seeds = config.get_config().get('securities')
        if securities_seeds:
            seed_securities(securities_seeds)
            log.info("Finished seeding securities.")
