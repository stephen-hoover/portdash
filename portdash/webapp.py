from flask import Blueprint

server_bp = Blueprint("main", __name__)


@server_bp.route("/")
def index():
    return
