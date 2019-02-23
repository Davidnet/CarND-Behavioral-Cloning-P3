import tensorflow as tf
import numpy as np

from .utils import add_coordinates
def pilot_relation_net(images, mode, params, conv_args ={}):

    training = mode == tf.estimator.ModeKeys.TRAIN

    net = images

    net = tf.layers.conv2d(net, 24, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 36, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 48, [5, 5], strides = 2, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = tf.layers.conv2d(net, 64, [3, 3], **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    net = add_coordinates(net)

    n_objects = np.prod(net.shape[1:-1])
    n_channels = net.shape[-1]
    
    net = tf.reshape(net, [-1, n_channels])

    net = tf.layers.dense(net, 200, **conv_args)
    net = tf.layers.batch_normalization(net, training=training)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net, rate = params.dropout, training=training)

    # aggregate relations
    n_channels = net.shape[1]
    net = tf.reshape(net, [-1, n_objects, n_channels])
    net = tf.reduce_max(net, axis = 1)

    # calculate global attribute
    net = tf.layers.dense(net, params.nbins)

    return dict(
        logits = net,
        probabilities = tf.nn.softmax(net),
    )



