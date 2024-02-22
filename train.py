import json

import matplotlib.pyplot as plt
import os
import re
import string
import tensorflow as tf
from keras import layers
from keras import losses


def make_train_val_datasets(train_dir, batch_size, seed):
    train_ds, val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='both',
        seed=seed)

    return train_ds, val_ds


def make_test_dataset(test_dir, batch_size):
    test_ds = tf.keras.utils.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size)
    
    return test_ds


def visualize_history(history):
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def compile_model(model):
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


def train_model(model, train_ds, val_ds, epochs):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)

    return history


def evaluate_model(model, test_ds):
    loss, accuracy = model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


# Create a data generator

batch_size = 32
seed = 42

# Make training, validation, and test datasets
train_dir = os.path.join('dataset', 'train')
test_dir = os.path.join('dataset', 'test')

raw_train_ds, raw_val_ds = make_train_val_datasets(train_dir, batch_size, seed)
raw_test_ds = make_test_dataset(test_dir, batch_size)

# Print class names
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

# Standardize the data
def custom_standardization(input_data):
    # convert to lower
    standardized = tf.strings.lower(input_data)

    # add spaces around punctuation
    for char in string.punctuation:
        # workaround, replacing with \ doesn't work 
        if char == '\\':
            continue
        standardized = tf.strings.regex_replace(standardized, re.escape(char), ' ' + char + ' ')

    # remove extra whitespace
    standardized = tf.strings.regex_replace(standardized, '[ +]', ' ')
 
    return standardized

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# performance configuration
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create the model

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

# Compile the model
compile_model(model)

# Train the model
history = train_model(model, train_ds, val_ds, epochs=30)

# Visualize the model
visualize_history(history)

# Evaluate the model
evaluate_model(model, test_ds)

# Export the model

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

export_model.save('model')