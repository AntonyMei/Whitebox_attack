import numpy as np
import tensorflow as tf

def dlr_loss(x, y, num_classes = 10):
    x_sort = tf.contrib.framework.sort(x, axis = 1)
    y_onehot = tf.one_hot(y, num_classes)
    ### TODO: adapt to the case when the point is already misclassified
    loss = -(x_sort[:, -1] - x_sort[:, -2]) / (x_sort[:, -1] - x_sort[:, -3] + 1e-12)
    print(str(x_sort))
    print(str(x))
    return loss


def dlr_loss_targeted(x, y, y_target, num_classes = 10):
    x_sort = tf.contrib.framework.sort(x, axis = 1)
    y_onehot = tf.one_hot(y, num_classes)
    y_target_onehot = tf.one_hot(y_target, num_classes)
    loss = -(tf.reduce_sum(x * y_onehot, axis = 1) - tf.reduce_sum(x * y_target_onehot, axis = 1)) / (
            x_sort[:, -1] - .5 * x_sort[:, -3] - .5 * x_sort[:, -4] + 1e-12)
    return loss


class CWLoss(Loss):
    ''' C&W loss. '''
    def __init__(self, model, c = 99999.0):
        self.model = model
        self.c = c


    def __call__(self, xs, ys):
        logits_mask = tf.one_hot(ys, self.model.n_class)
        logit_this = tf.reduce_sum(logits_mask * xs, axis = -1)
        logit_that = tf.reduce_max(xs - self.c * logits_mask, axis = -1)
        return logit_that - logit_this


class CW2Loss(Loss):
    ''' C&W loss. '''
    def __init__(self, model):
        self.model = model

    def __call__(self, xs, ys, ):
        xs = tf.nn.l2_normalize(xs, 1, 1e-10)
        logits_mask = tf.one_hot(ys, self.model.n_class)
        logit_this = tf.reduce_sum(logits_mask * xs, axis = -1)
        logit_that = tf.reduce_max(xs * (1. - logits_mask), axis = -1)
        return logit_that - logit_this


class CosLoss(Loss):
    ''' C&W loss. '''
    def __init__(self, model):
        self.model = model

    def __call__(self, xs, ys, ):
        logits_mask = tf.one_hot(ys, self.model.n_class)
        targets = self.c * logits_mask
        x_norm = tf.nn.l2_normalize(xs, 1, 1e-10)
        w_norm = tf.nn.l2_normalize(targets, 1, 1e-10)
        loss = 1. - tf.reduce_sum(tf.multiply(x_norm, w_norm), 1)
        return loss


class CELoss(Loss):
    ''' Cross entropy loss. '''


def __init__(self, model):
    self.model = model


def __call__(self, xs, ys):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = ys, logits = xs)
    return loss