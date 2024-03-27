import tensorflow as tf
from testing.privacypredict import FakeServer, FakeClient
from shared.text import make_vectorize_layer
from shared.common import *
import time
import numpy as np

class NonPrivacyPredictBenchmark:
    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary
        self.vectorize_layer = make_vectorize_layer(vocabulary)
    
    def predict(self, text):
        # Predict the text
        prediction = self.model.predict(text)
        
        # Pass the prediction through an activation function
        activation = tf.keras.activations.sigmoid
        prediction = activation(prediction)
        
        prediction = tf.reshape(prediction, (prediction.shape[0],))
        
        return prediction


class PrivacyPredictBenchmark:
    def __init__(self, model, vocabulary):
        self.model = model
        self.vocabulary = vocabulary
        
        # Server side initialization
        self.fake_server = FakeServer(model, vocabulary)
        
        # Send public params to client
        self.fake_client = FakeClient(*self.fake_server.get_public_params())
        
        # Send public params to server
        self.fake_server.set_client_params(*self.fake_client.get_public_params())
    
    def predict(self, text):
        # Encrypt the inputs
        encrypted_inputs, real_input_length = self.fake_client.get_encrypted_inputs(text)
        
        # Run the prediction on encrypted inputs
        encrypted_output = self.fake_server.predict(encrypted_inputs)
        
        # Decrypt the outputs
        prediction = self.fake_client.get_prediction(encrypted_output, real_input_length)
        
        return prediction
    

def run_benchmark(benchmark_class, model, vocabulary, test_dataset):
    # Initialize
    init_start_time = time.time()
    
    benchmark_instance = benchmark_class(model, vocabulary)
    
    init_end_time = time.time()
    init_total_time = init_end_time - init_start_time

    # Run predictions
    run_start_time = time.time()
    
    predictions = benchmark_instance.predict(test_dataset)
    
    run_end_time = time.time()
    run_total_time = run_end_time - run_start_time
    
    return init_total_time, run_total_time, np.array(predictions)


def prep_dataset(dataset_dir, vocabulary):
    # Get the dataset
    dataset = tf.keras.utils.text_dataset_from_directory(dataset_dir, seed=SEED, batch_size=BATCH_SIZE)
    
    # Get the texts and labels
    texts_and_labels = list(dataset.unbatch())
    texts = [text for text, label in texts_and_labels]
    labels = [label for text, label in texts_and_labels]
    
    # Vectorize text ahead of time
    vectorize_layer = make_vectorize_layer(vocabulary)
    texts = vectorize_layer(texts)

    return texts, labels


def calculate_accuracy(predictions, labels):
    predictions = np.array(predictions)
    
    # Convert predictions to binary
    predictions = np.round(predictions)
    
    # Calculate the accuracy
    accuracy = np.mean(predictions == labels)
    
    return accuracy


def benchmark(model, vocabulary, test_dataset_dir):
    # Get the test dataset
    texts, labels = prep_dataset(test_dataset_dir, vocabulary)

    # Run the non-privacy benchmark
    non_privacy_init_time, non_privacy_run_time, non_privacy_predictions = run_benchmark(NonPrivacyPredictBenchmark, model, vocabulary, texts)
    
    # Run the privacy benchmark
    privacy_init_time, privacy_run_time, privacy_predictions = run_benchmark(PrivacyPredictBenchmark, model, vocabulary, texts)
    
    # Calculate the accuracies using the labels
    non_privacy_accuracy = calculate_accuracy(non_privacy_predictions, labels)
    privacy_accuracy = calculate_accuracy(privacy_predictions, labels)
    
    print(f"Non-privacy init time: {non_privacy_init_time:.6f}s")
    print(f"Privacy init time:     {privacy_init_time:.6f}s")
    print(f"Non-privacy run time:  {non_privacy_run_time:.6f}s")
    print(f"Privacy run time:      {privacy_run_time:.6f}s")
    print(f"Non-privacy accuracy:  {non_privacy_accuracy:.6f}")
    print(f"Privacy accuracy:      {privacy_accuracy:.6f}")