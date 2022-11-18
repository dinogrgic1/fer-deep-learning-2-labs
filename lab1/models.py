import math
from typing import Union, Collection
from typing_extensions import TypeAlias

import tensorflow as tf

TFData: TypeAlias = Union[tf.Tensor, tf.Variable, float]

class GMModel:
    def __init__(self, K):
        self.K = K
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))

    @property
    def variables(self) -> Collection[TFData]:
        return self.mean, self.logvar, self.logpi

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)
        return 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)

    @tf.function
    def loss(self, data: TFData):
        return -tf.math.reduce_logsumexp(-((GMModel.neglog_normal_pdf(data, self.mean, self.logvar) + tf.math.reduce_logsumexp(self.logpi, 0) - self.logpi)), 1)
    
    def p_xz(self, x: TFData, k: int) -> TFData:
        var = tf.exp(self.logvar[k])
        return (1 / tf.math.sqrt(2 * math.pi * var)) * tf.exp(-0.5 * (((x - self.mean[k]) ** 2) / var))

    def p_x(self, x: TFData) -> TFData:
        pi = tf.nn.softmax(self.logpi)
        whole = tf.convert_to_tensor(0.0)
        for pi_, k in zip(pi, range(self.K)):
            values = pi_ * self.p_xz(x, k)
            whole = tf.add(values, whole)
        return whole
