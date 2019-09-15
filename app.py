import logging
import os

from portdash import create_app
from config import load_config

logging.basicConfig(level='DEBUG')
#logging.basicConfig(level=('DEBUG' if args.verbose else 'INFO'))
load_config(os.getenv("PORTDASH_CONF"))

server = create_app()
