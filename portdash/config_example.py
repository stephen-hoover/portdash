import os

###################################
# Data import configuration
# These need to be set by the user.
AV_API_KEY = 'SECRET'  # AlphaVantage API key

DATA_DIR = # Location of exported Acemoney files
INVESTMENT_TRANS_FNAME = os.path.join(DATA_DIR, 'investment_transactions.csv')
PORT_TRANSACTIONS = {
    'NAME': os.path.join(DATA_DIR, # Filename),
}
    
# Create a simulated portfolio from each symbol listed here.
SYMBOLS_TO_SIMULATE = ['SWPPX']

# Don't attempt to download these tickets
SKIP_SYMBOL_DOWNLOADS = {}

# Ignore these accounts from PORT_TRANSACTIONS
SKIP_ACCOUNTS = {}

##############################################################
# Cache directory -- this probably doesn't need to be changed.
CACHE_DIR = os.path.expanduser('~/.porttracker/cache')
os.makedirs(CACHE_DIR, exist_ok=True)
ETL_ACCTS = os.path.join(CACHE_DIR, "cached_accounts.pkl")
    
########################
# Dash app configuration
LINES = {'_total:1': 'Account Value',
         '365:7': 'Annual Rate of Return',
         '730:8': '2-year Average Annualized Rate of Return',
         'netcont:2': 'Net Contributions',
         'netgain:1.5': 'Net Gains',
         '_total_dist:3': 'Total Distributions',
         'contributions:4': 'Total Deposits',
         'withdrawals:5': 'Total Withdrawals',
         '28:6': '4-week Returns',
         }


WHICH_AXIS = {'_total:1': '$',
              'netcont:2': '$',
              'netgain:1.5': '$',
              '_total_dist:3': '$',
              'contributions:4': '$',
              'withdrawals:5': '$',
              '28:6': 'pct',
              '365:7': 'pct',
              '730:8': 'pct',
              }
