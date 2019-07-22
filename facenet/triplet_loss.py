import tensorflow as tf

def triplet_loss(y_true, y_pred, alpha=0.25):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    """
    Formula : sum((anchor-positive)^2 + (anchor - negative)^2 + alpha) from i=1 to N
    """

    positive_distance = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),axis=1)

    negative_distance = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), axis=1)

    loss = tf.reduce_sum(tf.maximum(tf.add(tf.subtract(positive_distance, negative_distance), alpha),0.0))

    return loss

    
