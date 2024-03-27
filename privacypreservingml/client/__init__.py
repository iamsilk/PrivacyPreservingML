import flask

def create_app():
    app = flask.Flask(__name__)
    return app