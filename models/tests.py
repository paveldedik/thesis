# -*- coding: utf-8 -*-

"""
Tools for Model Accuracy Estimation
===================================

"""

from . import tools, EloModel, PFAModel, PFAWithSpacing


class Test(object):

    def __init__(self, data, model_kwargs=None, train=True):
        self.model_kwargs = model_kwargs or {}
        self.train_set, self.test_set = self.split_data(data)

        if train:
            self.model = self.train_model()

    def retrain_with(self, model_kwargs=None):
        self.model_kwargs = model_kwargs or {}
        self.model = self.train_model()

    def split_data(self, data):
        raise NotImplementedError()

    def rmse(self):
        y_true = self.test_set['correct']
        y_pred = self.test_set.apply(self.model.predict, axis=1)
        return tools.rmse(y_true, y_pred)

    def auc(self):
        raise NotImplementedError()


class EloTest(Test):

    def train_model(self):
        model = EloModel(self.train_set, **self.model_kwargs)
        model.train()
        return model

    def split_data(self, data):
        data = tools.shuffle_data(tools.prior_data(data))
        return tools.split_data(data)


class PFATest(Test):

    def train_model(self):
        model = PFAModel(self.train_set, **self.model_kwargs)
        model.train()
        return model


class PFAWithSpacingTest(Test):

    def train_model(self):
        model = PFAWithSpacing(self.train_set, **self.model_kwargs)
        model.train()
        return model
