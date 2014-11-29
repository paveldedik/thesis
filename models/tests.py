# -*- coding: utf-8 -*-

"""
Tools for Model Accuracy Estimation
===================================

"""

from sklearn import metrics

from . import tools


class PrecisionResult(object):
    """Represents estimated accuracy of a model.

    :param value: Estimated accuracy.
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


class PrecisionTest(object):
    """Represents model precision test.

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
        train_set, test_set = self.model.split_data(self.data)
        self.model.train(train_set)

        self.y_true = test_set['is_correct']
        self.y_pred = test_set.apply(self.model.predict, axis=1)
        self.train_set_size = len(train_set)

    def rmse(self):
        """Estimates precision of the model using Root Mean Squared Error
        (RMSE) as metric.
        """
        result = tools.rmse(self.y_true, self.y_pred)
        return PrecisionResult(result, 'RMSE', self.train_set_size)

    def auc(self):
        """Estimates precision of the model using Area Under the Curve
        (AUC) as metric."""
        fpr, tpr, thresholds = \
            metrics.roc_curve(self.y_true, self.y_pred, pos_label=1)
        result = metrics.auc(fpr, tpr)
        return PrecisionResult(result, 'AUC', self.train_set_size)
