#!/usr/bin/env python3
import math
import numpy

def mean(data):

    return numpy.float64(sum(data)/len(data))


def stddev(data):

    m = mean(data)
    df = len(data) - 1

    s = 0
    for i in range(len(data)):
        s += math.pow(data[i] - m, 2)

    return numpy.float64(math.sqrt(s / df))


def covariance(v1, v2):

    if len(v1) != len(v2):
        raise ValueError('vectors of unequal length')

    m1 = mean(v1)
    m2 = mean(v2)

    s = 0
    for i in range(len(v1)):
        s += (v1[i] - m1) * (v2[i] - m2)

    return numpy.float64(s / len(v1))


class MultivariateNormal(object):

    def __init__(self, data):

        self.n_fields = len(data)
        self.n_datapoints = len(data[0])
        self.meanvector = numpy.zeros(shape=(self.n_fields,),
                                      dtype=numpy.float64)
        self.covmat = numpy.zeros(shape=(self.n_fields,
                                         self.n_fields),
                                  dtype=numpy.float64)
        self.__setvals(data)

    def __setvals(self, data):

        for i in range(self.n_fields):
            self.meanvector[i] = mean(data[i])

        for i in range(self.n_fields):
            for j in range(self.n_fields):
                self.covmat[i][j] = covariance(data[i], data[j])

    def likelihood(self, x):
        diff = x - self.meanvector
        diff = numpy.array([diff]).transpose()
        power = numpy.matmul(numpy.matmul(diff.transpose(), numpy.linalg.inv(self.covmat)), diff)[0][0] * (-1/2)
        exp = math.exp(power)
        const = 1 / math.sqrt(math.pow(2 * math.pi, self.n_fields) *
                              numpy.linalg.det(self.covmat))
        return const * exp
