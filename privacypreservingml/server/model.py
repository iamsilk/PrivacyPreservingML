import json
import tensorflow as tf
import numpy as np
import base64
import os

from eva import EvaProgram, Input, Output, evaluate
from eva.ckks import CKKSCompiler

from shared.common import *
from shared.serialization import eva_object_to_bytes, eva_object_from_bytes


model = None
vocabulary = None

embeddings = None

ckks_dense_program = None
ckks_params = None
ckks_signature = None

public_params_base64 = None
public_signature_base64 = None


def load_model(path):
    global model, embeddings
    model = tf.keras.models.load_model(path)
    embeddings = model.layers[0].embeddings.numpy().tolist()


def load_vocabulary(path):
    global vocabulary
    print('Loading vocabulary from', path)
    with open(path, "r") as f:
        vocabulary = json.load(f)
    print('Vocabulary loaded successfully', len(vocabulary))


def init_ckks_program():
    global ckks_dense_program, ckks_params, ckks_signature
    global public_params_base64, public_signature_base64
    
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
    ckks_dense_program, ckks_params, ckks_signature = compiler.compile(dense)
    
    public_params_base64 = base64.b64encode(eva_object_to_bytes(ckks_params)).decode()
    public_signature_base64 = base64.b64encode(eva_object_to_bytes(ckks_signature)).decode()


def predict(public_ctx, encrypted_inputs):
    # Predict the text
    encrypted_outputs = []
    for encrypted_input in encrypted_inputs:
        encrypted_output = public_ctx.execute(ckks_dense_program, encrypted_input)
        encrypted_outputs.append(encrypted_output)
    return encrypted_outputs


load_model(os.environ['MODEL_PATH'])
load_vocabulary(os.environ['VOCAB_PATH'])
init_ckks_program()