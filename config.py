import logging
import os
from typing import Any, Dict

import yaml

log = logging.getLogger(__name__)
INSTALL_DIR = os.path.dirname(__file__)

# Global state for app configuration, pre-set with defaults
CONF_STATE = {
    'data_dir': os.path.join(INSTALL_DIR, 'data'),
    'cache_dir': os.path.join(INSTALL_DIR, 'cache'),
    'etl_accts': os.path.join(INSTALL_DIR, 'cache', 'cached_accounts.pkl'),
    'skip_symbol_downloads': [],
    'skip_accounts': [],
    'symbols_to_simulate': [],

    ########################
    # Dash app configuration
    'lines': {'_total:1': 'Account Value',
              'netgain:1.5': 'Net Gains',
              'netcont:2': 'Net Contributions',
              '_total_dist:3': 'Total Distributions',
              'contributions:4': 'Total Deposits',
              'withdrawals:5': 'Total Withdrawals',
              '28:6': '4-week Returns',
              '365:7': 'Annual Rate of Return',
              '730:8': '2-year Average Annualized Rate of Return',
              },
    'which_axis': {'_total:1': '$',
                   'netgain:1.5': '$',
                   'netcont:2': '$',
                   '_total_dist:3': '$',
                   'contributions:4': '$',
                   'withdrawals:5': '$',
                   '28:6': 'pct',
                   '365:7': 'pct',
                   '730:8': 'pct',
                   },
}


def load_config(filename: str) -> None:
    with open(filename, 'rt') as _fin:
        conf_dict = yaml.load(_fin)
    update_config(conf_dict)


def update_config(config_dict) -> None:
    global CONF_STATE
    CONF_STATE.update(config_dict)

    for key in ['data_dir', 'cache_dir', 'etl_accts']:
        CONF_STATE[key] = os.path.expanduser(CONF_STATE[key])

    # Prepend the data directory to transaction filenames
    data_dir = CONF_STATE['data_dir']
    CONF_STATE['investment_transactions'] = os.path.join(
        data_dir, CONF_STATE['investment_transactions'])
    for key in CONF_STATE['account_transactions']:
        acct_files = CONF_STATE['account_transactions']
        acct_files[key] = os.path.join(data_dir, acct_files[key])

    if not CONF_STATE.get('av_api_key'):
        CONF_STATE['av_api_key'] = os.getenv('AV_API_KEY')


def conf(key: str) -> Any:
    return CONF_STATE[key]


def get_database_fname() -> str:
    """Get absolute filename from a possibly relative filename.
    Relative filenames refer to the package base directory.
    """
    fname = os.environ.get('DATABASE_URI') or conf('database_uri')
    if fname == os.path.basename(fname):
        fname = os.path.join(INSTALL_DIR, fname)
    return fname


def get_config() -> Dict:
    return CONF_STATE


def get_server_config():
    db_uri = get_database_fname()
    if not db_uri.startswith('sqlite'):
        db_uri = f'sqlite:///{db_uri}'
    log.debug(f'Setting database URI to {db_uri}')
    return {'SQLALCHEMY_DATABASE_URI': db_uri,
            'SQLALCHEMY_TRACK_MODIFICATIONS': False}
