import os
from flask import Flask

def create_app():
    # Create the Flask app
    app = Flask(__name__)
    
    # Register the blueprint
    from server.routes import bp
    app.register_blueprint(bp)
    
    # Test load the model, vocab, etc.
    import server.model
    
    return app