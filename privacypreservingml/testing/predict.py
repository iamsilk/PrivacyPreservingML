import tensorflow as tf
from shared.text import make_vectorize_layer

def predict_text(model, vocabulary, text):
    # Predict the text
    prediction = model.predict(text)
    
    # Pass the prediction through an activation function
    activation = tf.keras.activations.sigmoid
    prediction = activation(prediction)
    
    return prediction