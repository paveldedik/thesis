# -*- coding: utf-8 -*-

"""
Evaluation of Model Performance
===============================

"""

from __future__ import division

import numpy as np
import pandas as pd
from sklearn import metrics
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
        return metrics.roc_auc_score(self.observed, self.predicted)

    @tools.cached_property
    def off(self):
        """Difference between observed frequency of correct
        answers and average prediction.
        """
        return np.average(self.observed - self.predicted)

    def plot_roc(self):
        """Plots ROC curve (Receiver Operating Characteristic).
        """
        fpr, tpr, thresholds = \
            metrics.roc_curve(self.observed, self.predicted, pos_label=1)
        return plt.plot(fpr, tpr)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            'RMSE: {self.rmse}\n'
            'AUC: {self.auc}\n'
            'OFF: {self.off}\n'
            'Set Size: {self.size}'
        ).format(self=self)


class PerformanceTest(object):
    """Represents model performance test.

    :param model: Insance of the model to test.
    :type model: :class:`models.Model`
    :param data: Data to use for the test.
    :type data: :class:`pandas.DataFrame`
    :param split_data: Function that splits data into training set
        and test set.
    :type split_data: callable
    """

    def __init__(self, model, data, split_data=None):
        self.data = data
        self.model = model

        split_data = split_data or model.split_data
        self.train_set, self.test_set = split_data(data)

    def run(self):
        """Prepares training set, test set and trains the model.
        """
        self.model.train(self.train_set)

        self.test_values = pd.DataFrame({
            'observed': self.test_set['is_correct'],
            'predicted': self.test_set.apply(self.model.predict, axis=1),
        })
        self.train_values = pd.DataFrame(
            self.model.predictions,
            columns=['observed', 'predicted'],
        )

        self.test_result = PerformanceResult(
            self.test_values['observed'],
            self.test_values['predicted'],
        )
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
            'test': self.test_result,
            'train': self.train_result,
        }
