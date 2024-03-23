import tensorflow as tf
import string
from keras import layers
import re
from common import *

def standardize_text(input_text):
    # convert to lower
    standardized = tf.strings.lower(input_text)

    # add spaces around punctuation
    for char in string.punctuation:
        # workaround, replacing with \ doesn't work 
        if char == '\\':
            continue
        standardized = tf.strings.regex_replace(standardized, re.escape(char), ' ' + char + ' ')

    # remove extra whitespace
    standardized = tf.strings.regex_replace(standardized, r'\s+', ' ')
    
    # strip whitespace
    standardized = tf.strings.strip(standardized)
 
    return standardized


def make_vectorize_layer(vocabulary):
    return layers.TextVectorization(
        standardize=standardize_text,
        max_tokens=MAX_FEATURES,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH,
        vocabulary=vocabulary)