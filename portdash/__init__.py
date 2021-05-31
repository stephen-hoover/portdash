import dash
from flask import Flask
from flask.helpers import get_root_path

from config import get_server_config


def create_app():
    server = Flask(__name__)
    server.config.from_mapping(get_server_config())

    register_dashapps(server)
    register_extensions(server)

    return server


def register_dashapps(app):
    from portdash.dashapp.layout import layout
    from portdash.dashapp.callbacks import register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {
        "name": "viewport",
        "content": ("width=device-width, " "initial-scale=1, " "shrink-to-fit=no"),
    }

    dashapp = dash.Dash(
        __name__,
        server=app,
        url_base_pathname="/",
        assets_folder=get_root_path(__name__) + "/dashapp/assets/",
        meta_tags=[meta_viewport],
    )

    with app.app_context():
        dashapp.title = "Portfolio Dashboard"
        dashapp.layout = layout
        register_callbacks(dashapp)


def register_extensions(server):
    from portdash.extensions import db
    from portdash.extensions import migrate

    db.init_app(server)
    migrate.init_app(server, db)
