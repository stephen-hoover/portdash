# Portfolio Dashboard
Visualize value and returns over time in a stock portfolio

## To run

Setup:
```bash
source .envrc
pip install -r requirements.txt

flask db init
flask db migrate -m 'init'
flask db upgrade
```

Run!
```bash
PORTDASH_CONF=sample_config.yaml flask run
```