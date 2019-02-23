import tensorflow as tf
from tensorflow.python import autograph

from .model import pilot_relation_net

def model_fn(features, labels, mode, params):

    images = features["image"]

    predictions = pilot_relation_net(
        images,
        mode,
        params,
        conv_args=dict(
            kernel_regularizer=tf.contrib.layers.l2_regularizer(
                scale=params.l2_regularization
            )
        )
    )

    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=dict(
                serving_default=tf.estimator.export.PredictOutput(predictions)
            )
        )


    onehot_labels = get_onehot_labels(features["steering"], params)


    tf.losses.softmax_cross_entropy(
        onehot_labels = onehot_labels,
        logits = predictions["logits"],
        label_smoothing = params.label_smoothing,
        weights = get_weights(features["original_steering"], params)
    )

    loss = tf.losses.get_total_loss()

    labels = tf.argmax(onehot_labels, axis = 1)
    labels_pred = tf.argmax(predictions["logits"], axis = 1)

    if mode == tf.estimator.ModeKeys.EVAL:

        accuracy = tf.metrics.accuracy(labels, labels_pred)
        
        top_5_accuracy = tf.nn.in_top_k(
            predictions["logits"],
            labels,
            5,
        )
        top_5_accuracy = tf.cast(top_5_accuracy, tf.float32)
        top_5_accuracy = tf.metrics.mean(top_5_accuracy)

        return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions,
            loss = loss,
            eval_metric_ops = {
                "accuracy/top_5": top_5_accuracy,
                "accuracy/top_1": accuracy,
            }
        )

    if mode == tf.estimator.ModeKeys.TRAIN:
        
        with tf.name_scope("training"), tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            learning_rate = get_learning_rate(params)

            update = tf.contrib.opt.PowerSignOptimizer(
                learning_rate
            ).minimize(
                loss,
                global_step=tf.train.get_global_step()
            ) 

        # summaries
        accuracy = tf.contrib.metrics.accuracy(labels, labels_pred)

        top_5_accuracy = tf.nn.in_top_k(predictions["logits"], labels, 5)
        top_5_accuracy = tf.reduce_mean(tf.cast(top_5_accuracy, tf.float32))

        tf.summary.scalar("accuracy/top_1", accuracy)
        tf.summary.scalar("accuracy/top_5", top_5_accuracy)
        tf.summary.scalar("learning_rate", learning_rate)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=update,
        )


def get_weights(steering, params):


    ones = tf.ones_like(steering)
    zeros_weight = params.zeros_weight * ones

    weights = tf.where(
        tf.abs(steering) < 0.5,
        ones,
        ones * params.max_weight,
    )

    return tf.where(
        tf.equal(steering, 0.0),
        zeros_weight,
        weights,
    )


def get_onehot_labels(steering, params):
    
    label = tf.clip_by_value(steering, -1, 1)
    label = (label + 1.0) / 2.0
    label = label * (params.nbins - 1)

    label_upper = tf.ceil(label)
    label_lower = tf.floor(label)

    prob_upper = 1.0 - (label_upper - label)
    prob_upper = tf.cast(prob_upper, tf.float32)
    prob_upper = tf.expand_dims(prob_upper, 1)

    prob_lower = 1.0 - prob_upper

    onehot_upper = prob_upper * tf.one_hot(tf.cast(label_upper, tf.int32), params.nbins)
    onehot_lower = prob_lower * tf.one_hot(tf.cast(label_lower, tf.int32), params.nbins)

    onehot_labels = onehot_upper + onehot_lower

    return onehot_labels



@autograph.convert()
def get_learning_rate(params):
    
    global_step = tf.train.get_global_step()

    initial_learning_rate = params.learning_rate * params.batch_size / 128.0

    if global_step < params.cold_steps:
        learning_rate = params.cold_learning_rate
        learning_rate = tf.cast(learning_rate, tf.float32)
    
    elif global_step < params.cold_steps + params.warmup_steps:
        step = global_step - params.cold_steps
        p = step / params.warmup_steps
        
        learning_rate = initial_learning_rate * p + (1.0 - p) * params.cold_learning_rate
        learning_rate = tf.cast(learning_rate, tf.float32)

    else:
        step = global_step - (params.cold_steps + params.warmup_steps)
        learning_rate = tf.train.linear_cosine_decay(initial_learning_rate, step, params.decay_steps, beta = params.final_learning_rate)
        learning_rate = tf.cast(learning_rate, tf.float32)

    return learning_rate




