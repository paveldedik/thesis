# -*- coding: utf-8 -*-

"""
Evaluation Models
=================

"""

from __future__ import division

from datetime import datetime

import tools
import numpy as np


__all__ = (
    'EloModel',
    'EloResponseTime',
    'PFAModel',
    'PFATiming',
    'PFASpacing',
    'PFASpacingAlt',
)


class Model(object):
    """Abstract model class."""

    #: DateTime format of the field `inserted`.
    datetime_format = '%Y-%m-%d %H:%M:%S'

    def to_datetime(self, date_str):
        """Deserializes given datetime.

        :param date_str: DateTime given as string.
        :type date_str: str
        """
        return datetime.strptime(date_str, self.datetime_format)

    def respect_guess(self, prediction, options):
        """Updates prediction with respect to guessing paramter.

        :param prediction: Prediction calculated so far.
        :type prediction: float
        :param options: Number of options in the multiple-choice question.
        :type options: int
        """
        if options > 0:
            val = 1 / options
            return val + (1 - val) * prediction
        else:
            return prediction

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        raise NotImplementedError()

    def update(self, answer):
        """Performes an update of skills, difficulties or knowledge.

        :param answer: Asked question.
        :type answer: :class:`pandas.Series`
        """
        raise NotImplementedError()

    def train(self, data):
        """Trains the model on given data set.

        :param data: Data set on which to train the model.
        :type data: :class:`pandas.DataFrame`
        """
        raise NotImplementedError()

    @classmethod
    def split_data(cls, data, ratio=0.7):
        """Classmethod that splits data into training set and test set.

        :param data: The object containing data.
        :type data: :class:`pandas.DataFrame`.
        :param ratio: What portion of data to include in the training set
            and the test set. :obj:`0.5` means that the data will be
            distributed equaly.
        :type ratio: float
        """
        raise NotImplementedError()


class EloModel(Model):
    """Predicts correctness of answers using Elo Rating System.
    The model is parametrized with `alpha` and `beta`. These parameters
    affect the uncertainty function.
    """

    class _User(object):
        """Returns a user with given ID.

        :param user_id: ID of the user.
        :type user_id: int
        """
        def __init__(self, user_id):
            self.skill = 0.0
            self.number_of_answers = 0

    class _Place(object):
        """Returns a place with given ID.

        :param place_id: ID of the place.
        :type place_id: int
        """
        def __init__(self, place_id):
            self.difficulty = 0.0
            self.number_of_answers = 0

    def __init__(self, alpha=1, beta=0.05):
        self.alpha = alpha
        self.beta = beta

        self.init_model()

    def init_model(self):
        """Initializes two attributes of the model. Both attributes are
        dataframes. The first attribute represents difficulties of countries.
        The second attribute represents global knowledge of students.
        """
        self.places = tools.keydefaultdict(self._Place)
        self.users = tools.keydefaultdict(self._User)

    def uncertainty(self, n):
        """Uncertainty function. The purpose is to make each update on
        the model trained with larger data set less significant.

        :param n: Number of users or places.
        :type n: int
        """
        return self.alpha / (1 + self.beta * n)

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        user = self.users[question.user_id]
        place = self.places[question.place_id]

        prediction = tools.sigmoid(user.skill - place.difficulty)
        return self.respect_guess(prediction, question.number_of_options)

    def update(self, answer):
        """Updates skills of users and difficulties of places according
        to given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series`
        """
        user = self.users[answer.user_id]
        place = self.places[answer.place_id]

        shift = answer.is_correct - self.predict(answer)

        user.skill += self.uncertainty(user.number_of_answers) * shift
        place.difficulty -= self.uncertainty(place.number_of_answers) * shift

        user.number_of_answers += 1
        place.number_of_answers += 1

    def train(self, data):
        """Trains the model on given data set.

        :param data: Data set on which to train the model.
        :type data: :class:`pandas.DataFrame`
        """
        self.init_model()
        data = tools.first_answers(data)
        data.apply(self.update, axis=1)

    @classmethod
    def split_data(cls, data, ratio=0.7):
        """Classmethod that splits data into training set and test set.

        :param data: The object containing data.
        :type data: :class:`pandas.DataFrame`.
        :param ratio: What portion of data to include in the training set
            and the test set. :obj:`0.5` means that the data will be
            distributed equaly.
        :type ratio: float
        """
        data = tools.first_answers(data)
        return tools.split_data(data, ratio=ratio)


class EloResponseTime(EloModel):
    """Extension of the Elo model that takes response time of user
    into account.
    """

    def __init__(self, *args, **kwargs):
        self.phi = kwargs.pop('phi', 3)

        super(EloResponseTime, self).__init__(*args, **kwargs)

    def update(self, answer):
        """Updates skills of users and difficulties of places according
        to given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series`
        """
        user = self.users[answer.user_id]
        place = self.places[answer.place_id]

        level = tools.automaticity_level(answer.response_time)
        prob = (self.predict(answer) * self.phi + level) / (self.phi + 1)
        shift = answer.is_correct - prob

        user.skill += self.uncertainty(user.number_of_answers) * shift
        place.difficulty -= self.uncertainty(place.number_of_answers) * shift

        user.number_of_answers += 1
        place.number_of_answers += 1


class PFAModel(Model):
    """PFA model for estimation of current knowledge.

    :param gamma: The significance of the update when the student
        answered correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answered incorrectly.
    :type delta: float
    """

    class _Item(object):
        """Item representation.

        :param user_id: ID of the user.
        :type user_id: int
        :param place_id: ID of the place.
        :type place_id: int
        """

        def __init__(self, prior, user_id, place_id):
            self.user = prior.users[user_id]
            self.place = prior.places[place_id]
            self.knowledge = self.user.skill - self.place.difficulty

    def __init__(self, prior, gamma=3.4, delta=0.3):
        super(PFAModel, self).__init__()

        self.prior = prior
        self.gamma = gamma
        self.delta = delta

        self.init_model()

    def init_model(self):
        """Initializes attribute of the model that stores current
        knowledge of places for all students.
        """
        self.items = tools.keydefaultdict(
            lambda *args: self._Item(self.prior, *args)
        )

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        item = self.items[question.user_id, question.place_id]
        prediction = tools.sigmoid(item.knowledge)
        return self.respect_guess(prediction, question.number_of_options)

    def update(self, answer):
        """Performes update of current knowledge of a user based on the
        given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series`
        """
        item = self.items[answer.user_id, answer.place_id]
        prediction = self.predict(answer)

        if answer.is_correct:
            item.knowledge += self.gamma * (1 - prediction)
        else:
            item.knowledge += self.delta * (0 - prediction)

    def train(self, data):
        """Trains the model on given data set.

        :param data: Data set on which to train the model.
        :type data: :class:`pandas.DataFrame`
        """
        self.init_model()
        self.prior.train(data)
        data.apply(self.update, axis=1)

    @classmethod
    def split_data(self, data):
        """Classmethod that splits data into training set and test set.

        :param data: The object containing data.
        :type data: :class:`pandas.DataFrame`.
        """
        test_set = tools.last_answers(data)
        train_set = data[~data['id'].isin(test_set['id'])]

        return train_set, test_set


class PFATiming(PFAModel):
    """Alternative version of :class:`PFASpacing` which ignores
    spacing effect. Only forgetting is considered.
    """

    class _Item(PFAModel._Item):
        """Item representation.

        :param user_id: ID of the user.
        :type user_id: int
        :param place_id: ID of the place.
        :type place_id: int
        """

        def __init__(self, *args, **kwargs):
            self.practices = []
            super(PFATiming._Item, self).__init__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 3.4)
        kwargs.setdefault('delta', 0.3)

        time_effect = lambda t: 80 / t
        self.time_effect = kwargs.pop('time_effect_fun', time_effect)

        super(PFATiming, self).__init__(*args, **kwargs)

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        item = self.items[question.user_id, question.place_id]

        if item.practices:
            current_dt = self.to_datetime(question.inserted)
            last_answer = self.to_datetime(item.practices[-1])
            seconds = (current_dt - last_answer).total_seconds()
            time_effect = self.time_effect(seconds)
        else:
            time_effect = 0

        prediction = tools.sigmoid(item.knowledge + time_effect)
        return self.respect_guess(prediction, question.number_of_options)

    def update(self, answer):
        """Performes update of current knowledge of a user based on the
        given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series`
        """
        item = self.items[answer.user_id, answer.place_id]
        prediction = self.predict(answer)

        if answer.is_correct:
            item.knowledge += self.gamma * (1 - prediction)
        else:
            item.knowledge += self.delta * (0 - prediction)

        item.practices += [answer.inserted]


class PFASpacing(PFATiming):
    """Extended version of PFA that takes into account the effect of
    forgetting and spacing.

    :param gamma: The significance of the update when the student
        answers correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answers incorrectly.
    :type delta: float
    :param spacing_rate: The significance of the spacing effect. Lower
        values make the effect less significant. If the spacing rate
        is set to zero, the model is unaware of the spacing effect.
    :type spacing_rate: float
    :param decay_rate: The significance of the forgetting effect. Higher
        values of decay rate make the students forget the item faster
        and vice versa.
    :type decay_rate: float
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 3.4)
        kwargs.setdefault('delta', 0.3)

        self.spacing_rate = kwargs.pop('spacing_rate', 0)
        self.decay_rate = kwargs.pop('decay_rate', 0.2)

        self.tau = kwargs.pop('tau', 10)

        super(PFASpacing, self).__init__(*args, **kwargs)

    def _get_practices(self, current, prior):
        """Returns list of previous practices expresed as the number
        of seconds that passed between *current* practice and all
        the *prior* practices.

        :param current: Datetime of the current practice.
        :type place: string
        :param prior: List of datetimes of the prior practices.
        :type prior: list or :class:`numpy.array`
        """
        current_dt = self.to_datetime(current)
        prior_dts = [self.to_datetime(t) for t in prior]
        return [(current_dt - t).total_seconds() for t in prior_dts]

    def memory_strength(self, question):
        """Estimates memory strength of an item.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        item = self.items[question.user_id, question.place_id]
        practices = self._get_practices(question.inserted, item.practices)

        if len(practices) > 0:
            strength = tools.memory_strength(
                filter(lambda x: x > 0, practices),
                spacing_rate=self.spacing_rate,
                decay_rate=self.decay_rate,
            )
            return self.tau + strength

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        item = self.items[question.user_id, question.place_id]
        strength = self.memory_strength(question) or 0

        prediction = tools.sigmoid(item.knowledge + strength)
        return self.respect_guess(prediction, question.number_of_options)


class PFASpacingAlt(PFASpacing):
    """Alternative version of :class:`PFASpacing`.
    For description of the parameters of the model see the parent class.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 4.0)
        kwargs.setdefault('delta', 1.0)

        super(PFASpacingAlt, self).__init__(*args, **kwargs)

    def update(self, answer):
        """Performes update of current knowledge of a user based on the
        given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series`
        """
        item = self.items[answer.user_id, answer.place_id]
        prediction = self.predict(answer)

        if answer.is_correct:
            item.knowledge += self.gamma * (1 - prediction)
        else:
            item.knowledge += self.delta * (1 - prediction)

        item.practices += [answer.inserted]
