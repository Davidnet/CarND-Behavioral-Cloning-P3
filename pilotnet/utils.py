# Collections of util functions
import numpy as np
import tensorflow as tf 

from PIL import Image

def load_image(filename):
    try:
        with Image.open(filename) as img:
            return np.asarray(img)
    except IOError:
        return None


def add_coordinates(inputs, min_value = -1.0, max_value = 1.0):

    assert len(inputs.shape) > 2, "inputs must be of rank > 2"

    sample_base_shape = [ dim.value for dim in inputs.shape[1:-1] ]

    if not hasattr(min_value, "__iter__"):
        min_value = [ min_value ] * len(sample_base_shape)

    if not hasattr(max_value, "__iter__"):
        max_value = [ max_value ] * len(sample_base_shape)

    linspaces = [
        tf.linspace(start, stop, num)
        for start, stop, num in reversed(list(zip(min_value, max_value, sample_base_shape)))
    ]

    multiples = [tf.shape(inputs)[0]] + [1] * (len(inputs.shape) - 1)

    coords = tf.meshgrid(*linspaces)
    coords = tf.stack(coords, axis = -1)
    coords = tf.expand_dims(coords, axis = 0)
    coords = tf.tile(coords,multiples)

    return tf.concat([inputs, coords], axis = -1)