import base64
import os
import requests

import tensorflow as tf
from keras import layers

from eva.seal import generate_keys

from shared.common import *
from shared.text import make_vectorize_layer
from shared.serialization import eva_object_to_bytes, eva_object_from_bytes


api_base_url = os.environ['API_BASE_URL'].removesuffix('/')


vocab = None
embeddings = None
ckks_params = None
ckks_signature = None

public_ctx = None
secret_ctx = None
public_ctx_base64 = None

vectorize_layer = None
preparation_model = None


def fetch_vocab():
    r = requests.get(api_base_url + '/public/vocabulary')
    return r.json()


def fetch_embeddings():
    r = requests.get(api_base_url + '/public/embeddings')
    return r.json()


def fetch_ckks_params():
    r = requests.get(api_base_url + '/public/ckks/params')
    ckks_params_base64 = r.json()
    ckks_params_bytes = base64.b64decode(ckks_params_base64)
    ckks_params = eva_object_from_bytes(ckks_params_bytes)
    return ckks_params


def fetch_ckks_signature():
    r = requests.get(api_base_url + '/public/ckks/signature')
    ckks_signature_base64 = r.json()
    ckks_signature_bytes = base64.b64decode(ckks_signature_base64)
    ckks_signature = eva_object_from_bytes(ckks_signature_bytes)
    return ckks_signature


def make_preparation_model():
    # Prepare layers
    embedding_layer = layers.Embedding(MAX_FEATURES, EMBEDDING_DIM)
    embedding_layer.build()
    embedding_layer.load_own_variables({'0': embeddings})

    # Create the model
    preparation_model = tf.keras.Sequential([
        embedding_layer,
        layers.GlobalAveragePooling1D()
    ])
    
    # Compile the model
    preparation_model.compile()
    
    return preparation_model


def init():
    global vocab, embeddings, ckks_params, ckks_signature
    global public_ctx, secret_ctx, public_ctx_base64
    global vectorize_layer
    global preparation_model
    
    # Fetch server params
    vocab = vocab or fetch_vocab()
    embeddings = embeddings or fetch_embeddings()
    ckks_params = ckks_params or fetch_ckks_params()
    ckks_signature = ckks_signature or fetch_ckks_signature()
    
    # Create public and secret contexts
    if public_ctx is None or secret_ctx is None or public_ctx_base64 is None:
        public_ctx, secret_ctx = generate_keys(ckks_params)
        public_ctx_base64 = base64.b64encode(eva_object_to_bytes(public_ctx)).decode()
    
    # Make the vectorize layer
    vectorize_layer = vectorize_layer or make_vectorize_layer(vocab)
    
    # Make the preparation model
    preparation_model = preparation_model or make_preparation_model()


def get_encrypted_inputs(text):
        # Run the preparation model
        inputs = preparation_model.predict(text)
        
        real_input_length = len(inputs)
        if real_input_length < VECTOR_SIZE:
            padded_input_length = VECTOR_SIZE
        else:
            padded_input_length = 2**(len(inputs) - 1).bit_length()
        
        # Format the input for usage with the CKKS scheme
        formatted_inputs = { f"x_{i}": [] for i in range(EMBEDDING_DIM)}
        for i in range(real_input_length):
            for j in range(EMBEDDING_DIM):
                formatted_inputs[f"x_{j}"].append(inputs[i][j])
        
        # Pad the inputs
        for j in range(EMBEDDING_DIM):
            formatted_inputs[f"x_{j}"].extend([0] * (padded_input_length - real_input_length))
        
        # Split the inputs into batches
        formatted_input_batches = []
        for i in range(0, padded_input_length, VECTOR_SIZE):
            formatted_input_batch = { f"x_{j}": formatted_inputs[f"x_{j}"][i:i+VECTOR_SIZE] for j in range(EMBEDDING_DIM)}
            formatted_input_batches.append(formatted_input_batch)
        
        # Encrypt the inputs
        encrypted_inputs = []
        for formatted_input_batch in formatted_input_batches:
            encrypted_input_batch = public_ctx.encrypt(formatted_input_batch, ckks_signature)
            encrypted_inputs.append(encrypted_input_batch)
        
        return encrypted_inputs, len(inputs)


def submit_encrypted_inputs(encrypted_inputs):
    # Convert to bytes
    encrypted_inputs_bytes = [eva_object_to_bytes(encrypted_input) for encrypted_input in encrypted_inputs]
    
    # Convert to base64
    encrypted_inputs_base64 = [base64.b64encode(encrypted_input_bytes).decode() for encrypted_input_bytes in encrypted_inputs_bytes]
    
    # Send the encrypted inputs to the server
    r = requests.post(api_base_url + '/predict', json={
        'text': encrypted_inputs_base64,
        'public_ctx': public_ctx_base64
    })
    r.raise_for_status()
    
    # Get the encrypted outputs from the server
    encrypted_outputs_base64 = r.json()
    
    # Convert from base64
    encrypted_outputs_bytes = [base64.b64decode(encrypted_output_base64) for encrypted_output_base64 in encrypted_outputs_base64]
    
    # Convert from bytes
    encrypted_outputs = [eva_object_from_bytes(encrypted_output_bytes) for encrypted_output_bytes in encrypted_outputs_bytes]
    
    return encrypted_outputs


def get_prediction(encrypted_outputs, input_length):
    prediction = []
    
    # Decrypt the outputs
    for encrypted_output in encrypted_outputs:
        output = secret_ctx.decrypt(encrypted_output, ckks_signature)
        prediction.extend(output['y'])
    
    # Pass the prediction through an activation function
    activation = tf.keras.activations.sigmoid
    prediction = activation(prediction)
    
    # Truncate the prediction to the real input length
    prediction = prediction[:input_length]
    
    return prediction


def classify_text(text: list[str]) -> list[float]:
    # Initialize the client if it hasn't been initialized yet
    init()
    
    # Vectorize the text
    text = vectorize_layer(text)
    
    # Encrypt the inputs
    encrypted_inputs, input_length = get_encrypted_inputs(text)
    
    # Convert to bytes
    encrypted_inputs_bytes = [eva_object_to_bytes(encrypted_input) for encrypted_input in encrypted_inputs]
    
    # Send the encrypted inputs to the server
    encrypted_outputs = submit_encrypted_inputs(encrypted_inputs)
    
    # Decrypt the outputs
    predictions = get_prediction(encrypted_outputs, input_length)
    
    return predictions