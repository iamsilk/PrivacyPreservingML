import tensorflow as tf
from keras import layers
from text import standardize_text, make_vectorize_layer
from common import *
import numpy as np

from eva import EvaProgram, Input, Output
from eva.ckks import CKKSCompiler
from eva.seal import *

def predict_text(model, vocabulary, text):    
    ## SERVER-SIDE INITIALIZATION
    
    dense_layer = model.layers[4]
    
    dense = EvaProgram('Dense', vec_size=1)
    with dense:
        x = np.array([Input(f"x_{i}") for i in range(EMBEDDING_DIM)])
        w = dense_layer.kernel
        b = dense_layer.bias
        out = np.matmul(x, w)
        out = np.add(out, b)
        Output('y', out[0])

    dense.set_output_ranges(30)
    dense.set_input_scales(30)
    
    compiler = CKKSCompiler()
    compiled_dense, params, signature = compiler.compile(dense)
    
    ## CLIENT-SIDE INITIALIZATION
    
    public_ctx, secret_ctx = generate_keys(params)
    
    ## CLIENT-SIDE FUNCTION
    
    # Run the text through the preparation layers
    
    # Vectorization layer
    vectorize_layer = make_vectorize_layer(vocabulary)
    vectorized_text = vectorize_layer(text)
    
    inputs = tf.expand_dims(vectorized_text, 0)
    
    # Embedding layer
    embedding_layer = model.layers[0] # todo: create this layer
    inputs = embedding_layer(inputs)
    
    # GlobalAveragePooling1D layer
    pooling_layer = model.layers[2] # todo: create this layer
    inputs = pooling_layer(model.layers[1](inputs))
    
    # Encrypt the inputs
    
    inputs = { f"x_{i}": [inputs[0][i].numpy().item()] for i in range(EMBEDDING_DIM) }
    
    encrypted_inputs = public_ctx.encrypt(inputs, signature)
    
    # Send to server
    # todo: actually send this to the server
    
    ## SERVER-SIDE FUNCTION
    
    # Predict the text
    encrypted_output = public_ctx.execute(compiled_dense, encrypted_inputs)
    
    # Send to client
    # todo: actually send this to the client
    
    ## CLIENT-SIDE FUNCTION
    
    # Decrypt the outputs
    output = secret_ctx.decrypt(encrypted_output, signature)
    prediction = output['y'][0]
    
    # Pass the prediction through an activation function
    activation = tf.keras.activations.sigmoid
    prediction = activation(prediction)
    
    # Convert the prediction to a scalar
    prediction = prediction.numpy().item()
    
    return prediction