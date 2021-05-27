import numpy as np
import tensorflow as tf
from ares.attack.base import BatchAttack
from ares.attack.utils import get_xs_ph, get_ys_ph
import math
import sys
from sys import path
path.append(sys.path[0]+'/attacker')


class Attacker(BatchAttack):
    def __init__(self, model, batch_size, dataset, session):
        """ Based on ares.attack.bim.BIM, numpy version. """
        self.model, self.batch_size, self._session, self.dataset = model, batch_size, session, dataset
        if dataset == 'imagenet':
            self.class_num = 1000
        elif dataset == 'cifar10':
            self.class_num = 10
        # dataset == "imagenet" or "cifar10"
        # loss = CrossEntropyLoss(self.model)
        # placeholder for batch_attack's input
        self.xs_ph = get_xs_ph(model, batch_size)
        self.ys_ph = get_ys_ph(model, batch_size)
        self.logits = self.model.logits(self.xs_ph)

        label_mask = tf.one_hot(self.ys_ph,
                                self.class_num,
                                on_value=1.0,
                                off_value=0.0,
                                dtype=tf.float32)

        # softmax(logit) work better on other networks than cifar10, while logit work better on cifar10

        # SCW Loss
        self.softmax = tf.nn.softmax(self.logits)
        SCW_correct_logit = tf.reduce_sum(label_mask * self.softmax, axis=1)
        # 1e4: let the correct logit be small, correct implementation
        SCW_wrong_logit = tf.reduce_max((1 - label_mask) * self.softmax - 1e4 * label_mask, axis=1)
        self.SCW_margin_loss = -tf.nn.relu(SCW_correct_logit - SCW_wrong_logit + 50.)

        # CW Loss
        CW_correct_logit = tf.reduce_sum(label_mask * self.logits, axis=1)
        CW_wrong_logit = tf.reduce_max((1 - label_mask) * self.logits - 1e4 * label_mask,
                                            axis=1)
        self.CW_margin_loss = CW_wrong_logit - CW_correct_logit

        # gradients
        self.SCW_grad = tf.gradients(self.SCW_margin_loss, self.xs_ph)[0]
        self.CW_grad = tf.gradients(self.CW_margin_loss, self.xs_ph)[0]

        # random direction
        self.rand_direct = tf.Variable(np.zeros((self.batch_size, self.class_num)).astype(np.float32),
                                       name='rand_direct')
        self.rand_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.class_num])
        self.assign_op = self.rand_direct.assign(self.rand_placeholder)

        self.SCW_ODI_loss = tf.tensordot(self.softmax, self.rand_direct, axes=[[0, 1], [0, 1]])  # 用logits，不要纯用softmax
        self.SCW_grad_ODI = tf.gradients(self.SCW_ODI_loss, self.xs_ph)[0]

        self.CW_ODI_loss = tf.tensordot(self.logits, self.rand_direct,
                                             axes=[[0, 1], [0, 1]])
        self.CW_grad_ODI = tf.gradients(self.CW_ODI_loss, self.xs_ph)[0]

    def config(self, **kwargs):
        if 'magnitude' in kwargs:
            self.eps = kwargs['magnitude'] - 1e-8
            print("——————————————————————————————————————————————————————————————————————————————————————")
            self.alpha = self.eps
            self.ODI_alpha = self.eps

    def attack(self, start_times = 14, ODI_times = 2, iter_times = 5, rand = False,
               min_step = None, best_per = None, xs_lo = None, xs_hi = None,
               xs = None, xs_adv = None, ys = None,
               update_vector = None, replace_vector = None,
               gradODI = None, self_grad = None, ):
        for i in range(start_times):
            # best_per = xs #设置冷启动，热启动对imagenet基本无效

            if rand:  # 这个扰动很垃圾，没什么用
                # 这里设置热启动还是冷启动
                x = best_per + np.random.uniform(-self.eps, self.eps, xs.shape)
                x = np.clip(x, xs_lo, xs_hi)
                best_per = np.clip(x, self.model.x_min, self.model.x_max)

            rand_vector = np.random.uniform(-1.0, 1.0, (self.batch_size, self.class_num))
            self._session.run(self.assign_op,
                              feed_dict = {self.rand_placeholder: rand_vector.astype(np.float32)})
            for k in range(ODI_times):
                use_alpha = self.ODI_alpha
                # 使用热启动 ，要设置self.rand = false
                logits, grad = self._session.run([self.logits, gradODI],
                                                 feed_dict = {self.xs_ph: best_per,
                                                              self.ys_ph: ys})
                predict = np.argmax(logits, axis = 1)

                flag = (predict != ys)
                xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                # 攻击成功即停止更新
                update_vector[flag, :] = replace_vector[flag, :]

                grad_sign = np.sign(grad)

                xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                best_per = xxxx

            for j in range(iter_times):
                iteration = (j - 0) % iter_times
                # 余弦退火调整学习率
                use_alpha = min_step + (self.alpha - min_step) * (
                    1 + math.cos(math.pi * iteration / iter_times)) / 2
                # use_alpha = use_alpha / 4
                logits, grad = self._session.run([self.logits, self_grad],
                                                 feed_dict = {self.xs_ph: best_per, self.ys_ph: ys})

                predict = np.argmax(logits, axis = 1)

                flag = (predict != ys)
                xs_adv = xs_adv + (best_per - xs_adv) * update_vector
                # 攻击成功即停止更新
                update_vector[flag, :] = replace_vector[flag, :]

                # PGD
                grad_sign = np.sign(grad)
                xxxx = np.clip(best_per + use_alpha * grad_sign, xs_lo, xs_hi)
                xxxx = np.clip(xxxx, self.model.x_min, self.model.x_max)
                best_per = xxxx

    def batch_attack(self, xs, ys=None, ys_target=None):
        xs_lo, xs_hi = xs - self.eps, xs + self.eps
        # min_step = self.alpha / self.iteration
        # 攻击成功则停止更新
        update_vector = np.ones(xs.shape)
        replace_vector = np.zeros(xs.shape)
        xs_adv = xs
        # 利用logits判断是否攻击成功，失败的可以再重启
        important_logits_all = self._session.run(self.logits, feed_dict={self.xs_ph: xs,
                                                            self.ys_ph: ys})
        important_logits = np.max(important_logits_all)
        # important_logits_min = np.min(important_logits_all)

        if self.dataset == 'cifar10':
            if important_logits < 3.2 :
                print("MMC和feature_scatter定制")
                start_times = 14
                ODI_times = 2
                iter_times = 5
                min_step = self.alpha / 4
                best_per = xs_adv
                rand = False
                self.attack(start_times = start_times,
                            ODI_times = ODI_times,
                            iter_times = iter_times,
                            rand = rand,
                            min_step = min_step,
                            best_per = best_per,
                            xs_lo = xs_lo,
                            xs_hi = xs_hi,
                            xs = xs,
                            ys = ys,
                            update_vector = update_vector,
                            replace_vector = replace_vector,
                            gradODI = self.CW_grad_ODI,
                            self_grad = self.CW_grad)

                start_times = 14
                ODI_times = 2
                iter_times = 5
                min_step = self.alpha / 4
                best_per = xs_adv
                rand = False

                self.attack(start_times = start_times,
                            ODI_times = ODI_times,
                            iter_times = iter_times,
                            rand = rand,
                            min_step = min_step,
                            best_per = best_per,
                            xs_lo = xs_lo,
                            xs_hi = xs_hi,
                            xs = xs,
                            ys = ys,
                            update_vector = update_vector,
                            replace_vector = replace_vector,
                            gradODI = self.CW_grad_ODI,
                            self_grad = self.CW_grad)

            else:
                start_times = 14
                ODI_times = 2
                iter_times = 5
                min_step = self.alpha / 4
                best_per = xs_adv
                rand = False

                self.attack(start_times = start_times,
                            ODI_times = ODI_times,
                            iter_times = iter_times,
                            rand = rand,
                            min_step = min_step,
                            best_per = best_per,
                            xs_lo = xs_lo,
                            xs_hi = xs_hi,
                            xs = xs,
                            ys = ys,
                            update_vector = update_vector,
                            replace_vector = replace_vector,
                            gradODI = self.SCW_grad_ODI,
                            self_grad = self.SCW_grad)

        start_times = 5
        ODI_times = 2
        iter_times = 18
        min_step = self.alpha / 4
        best_per = xs_adv
        rand = False
        if self.dataset == 'imagenet':
            self.attack(start_times = start_times,
                        ODI_times = ODI_times,
                        iter_times = iter_times,
                        rand = rand,
                        min_step = min_step,
                        best_per = best_per,
                        xs_lo = xs_lo,
                        xs_hi = xs_hi,
                        xs = xs,
                        ys = ys,
                        update_vector = update_vector,
                        replace_vector = replace_vector,
                        gradODI = self.SCW_grad_ODI,
                        self_grad = self.SCW_grad)

        return xs_adv