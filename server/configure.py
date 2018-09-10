from flask import Flask
import os

FLASK_NAME = os.environ.get("FLASK_NAME")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")


def create_app():
    app = Flask(FLASK_NAME)
    app.secret_key = FLASK_SECRET_KEY
    return app


app = create_app()
