# -*- coding: utf-8 -*-

"""
Evaluation of Model Performance
===============================

"""

from __future__ import division

import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from . import tools


class PerformanceResult(object):
    """Represents estimated performance of a model.

    :param observed: Vector of observed corectness of answers.
    :type observed: iterable
    :param predicted: Vector of predicted corectness of answers.
        Must have the sime lenth as the vector containing observed
        answers.
    :type predicted: iterable
    """

    def __init__(self, observed, predicted):
        self.observed = observed
        self.predicted = predicted
        self.size = len(self.predicted)

    @tools.cached_property
    def rmse(self):
        """Estimates performance of the model using the Root Mean
         Squared Error (RMSE) as metric.
        """
        return tools.rmse(self.observed, self.predicted)

    @tools.cached_property
    def auc(self):
        """Estimates precision of the model using Area Under the Curve
        (AUC) as metric.
        """
        return sk.metrics.roc_auc_score(self.observed, self.predicted)

    @tools.cached_property
    def ll(self):
        """Log-likelihood, i.e. the logarithm of the likelihood.
        """
        return tools.log_likelihood(self.observed, self.predicted)

    @tools.cached_property
    def off(self):
        """Difference between observed frequency of correct
        answers and average prediction.
        """
        return np.average(self.predicted - self.observed)

    @tools.cached_property
    def accuracy(self):
        """Accuracy classification score."""
        return sk.metrics.accuracy_score(self.observed, self.predicted.round())

    @tools.cached_property
    def correct(self):
        """Number of correctly predicted answers."""
        return (self.observed == self.predicted.round()).sum()

    def plot_roc(self):
        """Plots ROC curve (Receiver Operating Characteristic).
        """
        fpr, tpr, thresholds = \
            sk.metrics.roc_curve(self.observed, self.predicted, pos_label=1)
        return plt.plot(fpr, tpr)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            'RMSE: {self.rmse}\n'
            'AUC: {self.auc}\n'
            'LL: {self.ll}\n'
            'OFF: {self.off}\n'
            'CORRECT: {self.correct}\n'
            'ACCURACY: {self.accuracy}\n'
            'Set Size: {self.size}'
        ).format(self=self)


class PerformanceTest(object):
    """Represents model performance test.

    :param model: Insance of the model to test.
    :type model: :class:`models.Model`
    :param data: Data to use for the test.
    :type data: :class:`pandas.DataFrame`
    :param split_data: Whether to split the data between
        test set and train set. Default is :obj:`False`.
    :type split_data: callable
    """

    def __init__(self, model, data, split_data=False):
        self.data = data
        self.model = model

        if split_data:
            self.train_set, self.test_set = model.split_data(data)
        else:
            self.train_set, self.test_set = data, None

    def run(self):
        """Prepares training set, test set and trains the model.
        """
        self.model.train(self.train_set)

        if self.test_set is not None:
            self.test_values = pd.DataFrame({
                'observed': self.test_set['is_correct'],
                'predicted': self.test_set.apply(self.model.predict, axis=1),
            })
            self.test_values = self.test_values[
                np.isfinite(self.test_values['predicted'])
            ]
            self.test_result = PerformanceResult(
                self.test_values['observed'],
                self.test_values['predicted'],
            )
        if self.train_set is not None:
            predictions = pd.DataFrame.from_dict(
                {'predicted': self.model.predictions},
            )
            self.train_set = pd.concat(
                [self.train_set.set_index(['id']), predictions], axis=1
            )
            self.train_values = pd.DataFrame({
                'observed': self.train_set['is_correct'],
                'predicted': self.train_set['predicted'],
            })
            self.train_values = self.train_values[
                np.isfinite(self.train_values['predicted'])
            ]
            self.train_result = PerformanceResult(
                self.train_values['observed'],
                self.train_values['predicted'],
            )

    @property
    def results(self):
        """Dictionary that contains results of test set
        and train set.
        """
        return {
            'test': getattr(self, 'test_result', None),
            'train': getattr(self, 'train_result', None),
        }
