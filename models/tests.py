# -*- coding: utf-8 -*-

"""
Evaluation of Model Performance
===============================

"""

from sklearn import metrics

from . import tools


class PerformanceResult(object):
    """Represents estimated performance of a model.

    :param value: Estimated performance.
    :type value: float
    :param method: Name of the used method (e.g. *RMES* or *AUC*).
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
    """

    def __init__(self, model, data):
        self.data = data
        self.model = model

    def run(self):
        """Prepares training set, test set and trains the model.
        """
        self.train_set, self.test_set = self.model.split_data(self.data)
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
