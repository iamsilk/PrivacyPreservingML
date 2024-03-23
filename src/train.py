import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import losses
from text import standardize_text, make_vectorize_layer, MAX_FEATURES

BATCH_SIZE = 32
SEED = 42
EMBEDDING_DIM = 16


def train_model(train_dataset_dir, epochs):
    # Split the dataset into training and validation
    raw_train_ds, raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dataset_dir,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='both',
        seed=SEED)
    
    # Get the vocabulary from the training dataset
    vocabulary = get_vocabulary_from_dataset(raw_train_ds)
    
    # Make the vectorize layer (also handles standardization of text)
    vectorize_layer = make_vectorize_layer(vocabulary)
    
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label
    
    # Vectorize (and standardize) the datasets
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    
    # Cache the training and validation datasets
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Create the model
    model = tf.keras.Sequential([
        layers.Embedding(MAX_FEATURES, EMBEDDING_DIM),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    
    # Compile the model
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)])
    
    # Train the model
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    
    return model, vocabulary


def evaluate_model(model, vocabulary, test_dataset_dir):
    # Get the test dataset
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        test_dataset_dir,
        batch_size=BATCH_SIZE)
    
    # Create the vectorize layer
    vectorize_layer = make_vectorize_layer(vocabulary)
    
    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label
    
    # Vectorize the test dataset
    test_ds = raw_test_ds.map(vectorize_text)
    
    # Cache the test dataset
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_ds)
    
    return loss, accuracy


def get_vocabulary_from_dataset(raw_train_ds):
    # Get training text for vectorization
    train_text = raw_train_ds.map(lambda text, label: text)
    
    # Make the vectorize layer (without vocabulary)
    vectorize_layer = make_vectorize_layer(None)
    
    # Adapt the vectorize layer to the training text
    vectorize_layer.adapt(train_text)
    
    # Export the vocabulary
    vocabulary = vectorize_layer.get_vocabulary()
    
    # Fix bug where there are extra ''s in the vocabulary
    if vocabulary[0] == '':
        vocabulary = [''] + [x for x in vocabulary[1:] if x != '']
    
    return vocabulary