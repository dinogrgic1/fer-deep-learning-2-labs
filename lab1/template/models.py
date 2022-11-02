import math
from typing import Union, Collection

import tensorflow as tf

TFData: Union[tf.Tensor, tf.Variable, float]

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
        raise NotImplementedError

    def p_xz(self, x: TFData, k: int) -> TFData:
        raise NotImplementedError

    def p_x(self, x: TFData) -> TFData:
        raise NotImplementedError
