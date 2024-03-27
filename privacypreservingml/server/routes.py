import base64

from flask import Blueprint, request, jsonify
from flask import current_app as app

from server.model import vocabulary, embeddings, public_params_base64, public_signature_base64
from server.model import predict
from shared.serialization import eva_object_to_bytes, eva_object_from_bytes


bp = Blueprint('routes', __name__)


@bp.route('/public/vocabulary', methods=['GET'])
def get_vocabulary():
    return jsonify(vocabulary)


@bp.route('/public/embeddings', methods=['GET'])
def get_embeddings():
    return jsonify(embeddings)


@bp.route('/public/ckks/params', methods=['GET'])
def get_ckks_params():
    return jsonify(public_params_base64)


@bp.route('/public/ckks/signature', methods=['GET'])
def get_ckks_signature():
    return jsonify(public_signature_base64)


@bp.route('/predict', methods=['POST'])
def run_predict():
    # Get the inputs
    public_ctx_base64 = request.json['public_ctx']
    encrypted_inputs_base64 = request.json['text']
    
    # Convert from base64
    public_ctx_bytes = base64.b64decode(public_ctx_base64)
    encrypted_inputs_bytes = [base64.b64decode(encrypted_input_base64) for encrypted_input_base64 in encrypted_inputs_base64]
    
    # Convert from bytes
    public_ctx = eva_object_from_bytes(public_ctx_bytes)
    encrypted_inputs = [eva_object_from_bytes(encrypted_input_bytes) for encrypted_input_bytes in encrypted_inputs_bytes]
    
    # Predict the text
    encrypted_outputs = predict(public_ctx, encrypted_inputs)
    
    # Convert to bytes
    encrypted_outputs_bytes = [eva_object_to_bytes(encrypted_output) for encrypted_output in encrypted_outputs]
    
    # Convert to base64
    encrypted_outputs_base64 = [base64.b64encode(encrypted_output_bytes).decode() for encrypted_output_bytes in encrypted_outputs_bytes]
    
    return jsonify(encrypted_outputs_base64)