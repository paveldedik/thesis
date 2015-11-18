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


def get_test_result(test_set, model):
    """Returns model's performance result on test data.

    :param test_set: Test set data.
    :param model: Model used for evaluation.
    """
    test_values = pd.DataFrame({
        'observed': test_set['is_correct'],
        'predicted': test_set.apply(model.predict, axis=1),
    })
    test_values = test_values[
        np.isfinite(test_values['predicted'])
    ]
    return PerformanceResult(
        test_values['observed'],
        test_values['predicted'],
    )


def get_train_result(train_set, model):
    """Returns model's performance result on train data.

    :param test_set: Train set data.
    :param model: Model used for evaluation.
    """
    predictions = pd.DataFrame.from_dict(
        {'predicted': model.predictions},
    )
    train_set = pd.concat(
        [train_set.set_index(['id']), predictions], axis=1
    )
    train_values = pd.DataFrame({
        'observed': train_set['is_correct'],
        'predicted': train_set['predicted'],
    })
    train_values = train_values[
        np.isfinite(train_values['predicted'])
    ]
    return PerformanceResult(
        train_values['observed'],
        train_values['predicted'],
    )


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

    def plot_dist(self):
        """Plots the distribution of true positives, true negatives,
        false positives and false negatives.
        """
        positives = []
        negatives = []

        joined = pd.concat([self.observed, self.predicted], axis=1)
        p = joined[self.observed == 1]
        n = joined[self.observed == 0]

        intervals = zip(np.arange(0, 0.99, 0.01), np.arange(0.01, 1, 0.01))

        for lower, upper in intervals:
            pc = len(p[(p['predicted'] > lower) & (p['predicted'] < upper)])
            nc = len(n[(n['predicted'] > lower) & (n['predicted'] < upper)])
            positives.append(pc)
            negatives.append(nc)

        X = [np.mean(interval) for interval in intervals]
        p1 = plt.plot(X, positives)
        p2 = plt.plot(X, negatives)

        plt.legend((p1[0], p2[0]), ('Positive samples', 'Negative samples'))
        plt.xlabel('Threshold')
        plt.ylabel('Number of samples')

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
            self.test_result = get_test_result(self.test_set, self.model)
        if self.train_set is not None:
            self.train_result = get_train_result(self.train_set, self.model)

    @property
    def results(self):
        """Dictionary that contains results of test set
        and train set.
        """
        return {
            'test': getattr(self, 'test_result', None),
            'train': getattr(self, 'train_result', None),
        }
