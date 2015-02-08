# -*- coding: utf-8 -*-

"""
Evaluation of Model Performance
===============================

"""

from __future__ import division

from sklearn import metrics
import matplotlib.pyplot as plt

from . import tools


class PerformanceResult(object):
    """Represents estimated performance of a model.

    :param value: Estimated performance.
    :type value: float
    :param method: Name of the used method (e.g. *RMSE* or *AUC*).
    :type method: string
    :param size: Size of the training set.
    :type size: int
    """

    def __init__(self, value, method, size):
        self.value = value
        self.method = method
        self.size = size

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return ('{self.method}: {self.value}\n'
                'Training Set Size: {self.size}').format(self=self)


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

        self.y_true = self.test_set['is_correct']
        self.y_pred = self.test_set.apply(self.model.predict, axis=1)

        self.test_set['prediction'] = self.y_pred

    def rmse(self):
        """Estimates performance of the model using the Root Mean
         Squared Error (RMSE) as metric.
        """
        result = tools.rmse(self.y_true, self.y_pred)
        return PerformanceResult(result, 'RMSE', len(self.train_set))

    def auc(self):
        """Estimates precision of the model using Area Under the Curve
        (AUC) as metric.
        """
        result = metrics.roc_auc_score(self.y_true, self.y_pred)
        return PerformanceResult(result, 'AUC', len(self.train_set))

    def plot_roc(self):
        """Plots ROC curve (Receiver Operating Characteristic).
        """
        fpr, tpr, thresholds = \
            metrics.roc_curve(self.y_true, self.y_pred, pos_label=1)
        return plt.plot(fpr, tpr)
