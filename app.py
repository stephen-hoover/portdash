import os

from portdash import create_app
from config import load_config

load_config(os.getenv("PORTDASH_CONF"))

server = create_app()
