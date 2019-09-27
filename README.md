# Portfolio Dashboard
Visualize value and returns over time in a stock portfolio

## To run

Setup:
```bash
source .envrc
pip install -r requirements.txt

PORTDASH_CONF=sample_config.yaml flask db init
PORTDASH_CONF=sample_config.yaml flask db migrate -m 'init'
PORTDASH_CONF=sample_config.yaml flask db upgrade
```

See `migrations/alembic.ini` for logger settings.

Next, fill the database. Start by seeding from your config file.
```bash
python portdash/apis/seeding.py -c config_filename.yaml
```

Read CSVs of transactions and update quotes from Alpha Vantage
```bash
python portdash/acemoney.py -c config_filename.yaml --quotes
```

Run!
```bash
PORTDASH_CONF=sample_config.yaml flask run
```
