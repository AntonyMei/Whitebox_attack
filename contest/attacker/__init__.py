import numpy as np
import math
import attacker.util as util
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph, maybe_to_array
from ares.loss import CrossEntropyLoss

# base PGD attacker
class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        ''' Based on ares.attack.bim.BIM '''
        self.model, self.batch_size, self._session = model, batch_size, session
        model_loss = CrossEntropyLoss(self.model)
        # Checkpoints for alpha
        self.iteration_count = 100
        self.checkpoint_count = 9  # 9 checkpoints, 8 intervals
        self.p = [0, 0.22]
        self.checkpoint = []
        # Counters for alpha
        self.local_alpha = 0
        self.rho = 0.75
        self.current_iteration = 0
        self.better_f_count = 0
        self.last_loss = 0
        self.last_alpha = 0
        self.last_max_f = 0
        self.loss = 0

        # Init checkpoints for alpha
        for i in range(self.checkpoint_count - 2):
            j = i + 1
            self.p.append(self.p[j] + max(self.p[j] - self.p[j - 1] - 0.03, 0.06))
        for position in self.p:
            self.checkpoint.append(math.ceil(position * self.iteration_count))

        # dataset == "imagenet" or "cifar10"
        # loss = CrossEntropyLoss(self.model)

        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)

        # flatten shape of xs_ph
        xs_flatten_shape = (batch_size, np.prod(self.model.x_shape))

        # store xs and ys in variables to reduce memory copy between tensorflow and python
        # variable for the original example with shape of (batch_size, D)
        self.xs_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))

        # variable for labels
        self.ys_var = tf.Variable(tf.zeros(shape=(batch_size,), dtype=self.model.y_dtype))

        # variable for the (hopefully) adversarial example with shape of (batch_size, D)
        self.xs_adv_var = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))

        # magnitude
        self.eps_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.eps_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))

        # step size
        self.alpha_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        self.alpha_var = tf.Variable(tf.zeros((self.batch_size,), dtype=self.model.x_dtype))

        # expand dim for easier broadcast operations
        eps = tf.expand_dims(self.eps_var, 1)
        alpha = tf.expand_dims(self.alpha_var, 1)

        # TODO: check momentum size(wwh)
        # self.momentum_ph = tf.placeholder(self.model.x_dtype, (self.batch_size,))
        # self.momentum_var = tf.Variable(tf.zeros((self.batch_size,), dtype = self.model.x_dtype))
        momentum = tf.constant(0.75)

        # TODO: check initialize
        self.setup_xs = [self.xs_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape)),
                         self.xs_adv_var.assign(tf.reshape(self.xs_ph, xs_flatten_shape))]
        self.setup_ys = self.ys_var.assign(self.ys_ph)
        # 2 losses
        # TODO: alter loss to dlr_loss
        self.xs_adv_model = tf.reshape(self.xs_adv_var, (batch_size, *self.model.x_shape))
        self.loss = model_loss(self.xs_adv_model, self.ys_var)
        # self.loss = util.dlr_loss(self.xs_adv_model, self.ys_var, self.model.n_class)
        # grad shape: [batch_size, D]
        grad = tf.gradients(self.loss, self.xs_adv_var)[0]

        xs_lo, xs_hi = self.xs_var - eps, self.xs_var + eps

        # clip by max l_inf magnitude of adversarial noise
        x1 = tf.clip_by_value(self.xs_adv_var + alpha * grad, xs_lo, xs_hi)
        self.update_xs_adv_step1 = self.xs_adv_var.assign(x1)

        # update the adversarial example
        # TODO: alter loss to dlr_loss
        self.xs_adv_model = tf.reshape(self.xs_adv_var, (batch_size, *self.model.x_shape))
        loss1 = model_loss(self.xs_adv_model, self.ys_var)
        # loss1 = util.dlr_loss(self.xs_adv_model, self.ys_var, self.model.n_class)
        grad = tf.gradients(loss1, self.xs_adv_var)[0]

        x2 = tf.clip_by_value(self.xs_adv_var + alpha * grad, xs_lo, xs_hi)
        self.update_xs_adv_step2 = self.xs_adv_var.assign(x2)

        self.loss_max = tf.reduce_max([self.loss, loss1])
        # TODO: loss!!!
        print(self.loss)
        print(loss1)
        for i in range(batch_size):
            self.x_max = tf.cond(self.loss[i] < loss1[i], lambda: self.setup_xs, lambda: x1)

        # start iterate
        # calculate loss' gradient with relate to the adversarial example
        # grad.shape == (batch_size, D)
        self.xs_adv_model = tf.reshape(self.xs_adv_var, (batch_size, *self.model.x_shape))
        self.loss = model_loss(self.xs_adv_model, self.ys_var) # TODO: alter to dlr_loss
        grad = tf.gradients(self.loss, self.xs_adv_var)[0]

        # update the adversarial example
        xs_lo, xs_hi = self.xs_var - eps, self.xs_var + eps

        # clip by max l_inf magnitude of adversarial noise
        self.xs_adv_var_temp1 = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        self.xs_adv_var_temp2 = tf.Variable(tf.zeros(shape=xs_flatten_shape, dtype=self.model.x_dtype))
        xs_adv_next_temp1 = tf.clip_by_value(self.xs_adv_var + alpha * grad, xs_lo, xs_hi)  # wwh: z_k+1
        xs_adv_next_temp2 = tf.clip_by_value(self.xs_adv_var + momentum * (xs_adv_next_temp1 - self.xs_adv_var) + (1 - momentum), xs_lo, xs_hi)

        loss_next = util.dlr_loss(xs_adv_next_temp2, self.ys_var, self.model.n_class)
        (self.x_max, self.loss_max) = tf.cond(loss_next[0] > self.loss_max[0], lambda: (xs_adv_next_temp2, loss_next), lambda: (self.x_max, self.loss_max))

        # clip by (x_min, x_max)
        xs_adv_next = tf.clip_by_value(xs_adv_next_temp2, self.model.x_min, self.model.x_max)

        self.update_xs_adv_step = self.xs_adv_var.assign(xs_adv_next)
        self.config_eps_step = self.eps_var.assign(self.eps_ph)
        self.config_alpha_step = self.alpha_var.assign(self.alpha_ph)

        # TODO: check the number of iterations
        self.iteration = 100

    # In each step, call this function to calculate alpha and feed into alpha_ph
    def calculate_alpha(self, ):
        self.current_iteration += 1

        # update loss
        self.better_f_count = tf.cond(self.loss[0] > self.last_loss[0], lambda: self.better_f_count + 1, lambda: self.better_f_count)
        self.last_loss = self.loss

        # update alpha
        for j in range(self.checkpoint_count):
            if self.current_iteration == self.checkpoint[j]:
                # check condition 1
                condition1 = False
                interval_length = self.checkpoint[j] - self.checkpoint[j - 1]
                hit_count = interval_length * self.rho
                if self.better_f_count < hit_count:
                    condition1 = True
                # check condition 2
                condition2 = False
                if self.last_alpha == self.local_alpha and self.last_max_f == self.loss_max:
                    condition2 = True
                # update parameters
                self.better_f_count = 0
                self.last_alpha = self.local_alpha
                self.last_max_f = self.loss_max
                # halve alpha if necessary
                if condition1 or condition2:
                    self.local_alpha /= 2

                break

        return self.local_alpha

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-6
            eps = maybe_to_array(self.eps, self.batch_size)
            self._session.run(self.config_eps_step, feed_dict={self.eps_ph: eps})
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: self.eps * 2})
            self.local_alpha = self.eps * 2

    def batch_attack(self, xs, ys=None, ys_target=None):
        self._session.run(self.setup_xs, feed_dict={self.xs_ph: xs})
        self._session.run(self.setup_ys, feed_dict={self.ys_ph: ys})
        self._session.run(self.x_max, feed_dict={self.xs_ph: xs, self.ys_ph: ys})
        self._session.run(self.loss_max, feed_dict={self.xs_ph: xs, self.ys_ph: ys})
        self._session.run(self.update_xs_adv_step1)
        print("done 1")
        self._session.run(self.update_xs_adv_step2)
        print("done 2")
        # wwh: range -1 because of computation of x_max above
        for _ in range(self.iteration - 1):
            self._session.run(self.config_alpha_step, feed_dict={self.alpha_ph: self.calculate_alpha()})
            self._session.run(self.loss_max, feed_dict={})
            self._session.run(self.update_xs_adv_step)
        return self._session.run(self.xs_adv_model)
