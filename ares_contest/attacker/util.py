import numpy as np
import tensorflow.compat.v1 as tf

def dlr_loss(x, y, num_classes = 10):
    x_sort = tf.contrib.framework.sort(x, axis = 1)
    y_onehot = tf.one_hot(y,  num_classes)
    ### TODO: adapt to the case when the point is already misclassified
    loss = -(x_sort[:, -1] - x_sort[:, -2]) / (x_sort[:, -1] - x_sort[:, -3] + 1e-12)

    return loss


def dlr_loss_targeted(x, y, y_target, num_classes = 10):
    x_sort = tf.contrib.framework.sort(x, axis = 1)
    y_onehot = tf.one_hot(y, num_classes)
    y_target_onehot = tf.one_hot(y_target, num_classes)
    loss = -(tf.reduce_sum(x * y_onehot, axis = 1) - tf.reduce_sum(x * y_target_onehot, axis = 1)) / (
            x_sort[:, -1] - .5 * x_sort[:, -3] - .5 * x_sort[:, -4] + 1e-12)
    return loss