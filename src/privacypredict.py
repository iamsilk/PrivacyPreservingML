import tensorflow as tf
from keras import layers
from text import standardize_text, make_vectorize_layer
from common import *
import numpy as np

from eva import EvaProgram, Input, Output
from eva.ckks import CKKSCompiler
from eva.seal import *


class FakeServer:
    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary
    
        dense_layer = model.layers[4]
        
        dense = EvaProgram('Dense', vec_size=1)
        with dense:
            x = np.array([Input(f"x_{i}") for i in range(EMBEDDING_DIM)])
            out = np.matmul(x, dense_layer.kernel)
            out = np.add(out, dense_layer.bias)
            Output('y', out[0])

        dense.set_output_ranges(30)
        dense.set_input_scales(30)
        
        compiler = CKKSCompiler()
        compiled_dense, params, signature = compiler.compile(dense)
        
        self.compiled_dense = compiled_dense
        self.params = params
        self.signature = signature

    def get_public_params(self):
        return self.params, self.signature, self.vocabulary, self.model.layers[0].embeddings.numpy()
    
    def set_client_params(self, public_ctx):
        self.public_ctx = public_ctx
    
    def predict(self, encrypted_inputs):
        # Predict the text
        encrypted_output = self.public_ctx.execute(self.compiled_dense, encrypted_inputs)
        
        return encrypted_output
    

class FakeClient:
    def __init__(self, params, signature, vocabulary, embeddings):
        self.params = params
        self.signature = signature
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        
        # Generate keys
        self.public_ctx, self.secret_ctx = generate_keys(self.params)
        
        # Prepare layers
        self.layers = [
            make_vectorize_layer(self.vocabulary),
            self.layer_expand_dims,
            self.layer_embedding,
            layers.GlobalAveragePooling1D()
        ]
    
    def get_public_params(self):
        return (self.public_ctx, )
    
    def layer_expand_dims(self, inputs):
        return tf.expand_dims(inputs, 0)
    
    def layer_embedding(self, inputs):
        return np.take(self.embeddings, inputs, axis=0)
    
    def get_encrypted_inputs(self, text):        
        # Preparation layers
        inputs = text
        
        for layer in self.layers:
            inputs = layer(inputs)
        
        # Encrypt the inputs
        inputs = { f"x_{i}": [inputs[0][i].numpy().item()] for i in range(EMBEDDING_DIM) }
        encrypted_inputs = self.public_ctx.encrypt(inputs, self.signature)
        
        return encrypted_inputs
    
    def get_prediction(self, encrypted_output):
        # Decrypt the outputs
        output = self.secret_ctx.decrypt(encrypted_output, self.signature)
        prediction = output['y'][0]
        
        # Pass the prediction through an activation function
        activation = tf.keras.activations.sigmoid
        prediction = activation(prediction)
        
        # Convert the prediction to a scalar
        prediction = prediction.numpy().item()
        
        return prediction


def predict_text(model, vocabulary, text):    
    # Server side initialization
    fake_server = FakeServer(model, vocabulary)
    
    # Send public params to client
    fake_client = FakeClient(*fake_server.get_public_params())
    
    # Send public params to server
    fake_server.set_client_params(*fake_client.get_public_params())
    
    # Encrypt the inputs
    encrypted_inputs = fake_client.get_encrypted_inputs(text)
    
    # Run the prediction on encrypted inputs
    encrypted_output = fake_server.predict(encrypted_inputs)
    
    # Decrypt the outputs
    prediction = fake_client.get_prediction(encrypted_output)
    
    return prediction