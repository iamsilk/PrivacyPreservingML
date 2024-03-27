from flask import Flask


def create_app():
    # Create the Flask app
    app = Flask(__name__)
    
    # Register the blueprint
    from client.routes import bp
    app.register_blueprint(bp)
    
    return app