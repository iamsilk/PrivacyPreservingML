import tensorflow as tf
from keras import layers
from shared.text import standardize_text, make_vectorize_layer
from shared.common import *
import numpy as np

from eva import EvaProgram, Input, Output
from eva.ckks import CKKSCompiler
from eva.seal import *


class FakeServer:
    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary
    
        dense_layer = model.layers[4]
        
        dense = EvaProgram('Dense', vec_size=VECTOR_SIZE)
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
        encrypted_outputs = []
        for encrypted_input in encrypted_inputs:
            encrypted_output = self.public_ctx.execute(self.compiled_dense, encrypted_input)
            encrypted_outputs.append(encrypted_output)
        return encrypted_outputs
    

class FakeClient:
    def __init__(self, params, signature, vocabulary, embeddings):
        self.params = params
        self.signature = signature
        self.vocabulary = vocabulary
        self.embeddings = embeddings
        
        # Generate keys
        self.public_ctx, self.secret_ctx = generate_keys(self.params)
        
        # Create text vectorization layer
        self.vectorize_layer = make_vectorize_layer(self.vocabulary)

        # Prepare layers
        embedding_layer = layers.Embedding(MAX_FEATURES, EMBEDDING_DIM)
        embedding_layer.build()
        embedding_layer.load_own_variables({'0': self.embeddings})

        self.preparation_model = tf.keras.Sequential([
            embedding_layer,
            layers.GlobalAveragePooling1D()
        ])
        
        self.preparation_model.compile()
    
    def get_public_params(self):
        return (self.public_ctx, )
    
    def layer_embedding(self, inputs):
        return np.take(self.embeddings, inputs, axis=0)
    
    def vectorize_text(self, text):
        return self.vectorize_layer(text)
    
    def get_encrypted_inputs(self, text):
        # Run the preparation model
        inputs = self.preparation_model.predict(text)
        
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
            encrypted_input_batch = self.public_ctx.encrypt(formatted_input_batch, self.signature)
            encrypted_inputs.append(encrypted_input_batch)
        
        return encrypted_inputs, len(inputs)
    
    def get_prediction(self, encrypted_outputs, input_length):
        prediction = []
        
        # Decrypt the outputs
        for encrypted_output in encrypted_outputs:
            output = self.secret_ctx.decrypt(encrypted_output, self.signature)
            prediction.extend(output['y'])
        
        # Pass the prediction through an activation function
        activation = tf.keras.activations.sigmoid
        prediction = activation(prediction)
        
        # Truncate the prediction to the real input length
        prediction = prediction[:input_length]
        
        return prediction


def predict_text(model, vocabulary, text):
    # Server side initialization
    fake_server = FakeServer(model, vocabulary)
    
    # Send public params to client
    fake_client = FakeClient(*fake_server.get_public_params())
    
    # Send public params to server
    fake_server.set_client_params(*fake_client.get_public_params())
    
    # Encrypt the inputs
    encrypted_inputs, real_input_length = fake_client.get_encrypted_inputs(text)
    
    # Run the prediction on encrypted inputs
    encrypted_outputs = fake_server.predict(encrypted_inputs)
    
    # Decrypt the outputs
    prediction = fake_client.get_prediction(encrypted_outputs, real_input_length)
    
    return prediction