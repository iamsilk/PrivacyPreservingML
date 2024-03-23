import tensorflow as tf
from text import standardize_text, make_vectorize_layer

def predict_text(model, vocabulary, text):
    # Standardize and vectorize the text
    vectorize_layer = make_vectorize_layer(vocabulary)
    vectorized_text = vectorize_layer(text)
    
    # Predict the text
    prediction = model.predict(tf.expand_dims(vectorized_text, 0))
    
    # Pass the prediction through an activation function
    activation = tf.keras.activations.sigmoid
    prediction = activation(prediction)
    
    # Convert the prediction to a scalar
    prediction = prediction.numpy().item()
    
    return prediction