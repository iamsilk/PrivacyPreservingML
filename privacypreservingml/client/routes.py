from flask import Blueprint, request, render_template
from client.model import classify_text


bp = Blueprint('routes', __name__)


@bp.route('/', methods=['GET', 'POST'])
def index():
    # If the request is a POST request, classify the text
    if request.method == 'POST':
        # Get the text
        text = request.form['text']
        
        # Run the prediction
        prediction = classify_text([text]).numpy()[0]
        
        # Return the prediction
        return render_template('index.html', predicted=True, text=text, prediction=round(prediction*100, 1))

    return render_template('index.html', predicted=False)