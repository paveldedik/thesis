# -*- coding: utf-8 -*-

"""
Evaluation Models
=================

"""

import pandas as pd
from datetime import datetime

import tools


class DataSet(object):
    """Most important data structure which somewhat simplifies working
    with models used for the estimation of prior and current knowledge
    of students.

    :param data: DataFrame with user's answers.
    :type data: :class:`pandas.DataFrame`.
    """

    datetime_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, data):
        self.data = data

        self.difficulties = pd.DataFrame(
            columns=['difficulty'],
            index=pd.Index([], name='place_asked')
        )
        self.skills = pd.DataFrame(
            columns=['skill'],
            index=pd.Index([], name='user')
        )
        self.knowledge = pd.DataFrame(
            columns=['knowledge'],
            index=pd.MultiIndex([[], []], [[], []], names=['user', 'place'])
        )

    @tools.cached_property
    def places(self):
        """Returns all unique places."""
        return self.data['place_asked'].unique()

    @tools.cached_property
    def users(self):
        """Returns all unique users."""
        return self.data['user'].unique()

    @tools.cached_property
    def prior(self):
        """Returns prior answers for all users."""
        return tools.prior_data(self.data)

    @tools.cached_property
    def merged(self):
        """Returns estimated difficulties and skills in one dataframe."""
        return pd.merge(
            pd.merge(self.data, self.difficulties.reset_index(),
                     on='place_asked'),
            self.skills.reset_index(), on='user'
        )

    def get_skill(self, user):
        """Returns estimated skill of given user.

        :param user: ID of the user.
        :type user: int
        """
        if user in self.skills.index:
            return self.skills.get_value(user, 'skill')
        else:
            return 0

    def set_skill(self, user, value):
        """Sets skill of given user.

        :param user: ID of the user.
        :type user: int
        :param value: Value of the estimated skill.
        :type skill: float
        """
        self.skills.set_value(user, 'skill', value)

    def get_difficulty(self, place):
        """Returns estimated difficulty of given place.

        :param place: ID of the place.
        :type place: int
        """
        if place in self.difficulties.index:
            return self.difficulties.get_value(place, 'difficulty')
        else:
            return 0

    def set_difficulty(self, place, value):
        """Sets difficulty of given place.

        :param place: ID of the place.
        :type place: int
        :param value: Value of the estimated difficulty.
        :type skill: float
        """
        self.difficulties.set_value(place, 'difficulty', value)

    def get_knowledge(self, user, place):
        """Returns user's knowledge of the given place.

        :param user: ID of the user.
        :type user: int
        :param place: ID of the place.
        :type place: int
        """
        if (user, place) in self.knowledge.index:
            return self.knowledge.get_value((user, place), 'knowledge')
        else:
            skill = self.get_skill(user)
            difficulty = self.get_difficulty(place)
            return skill - difficulty

    def set_knowledge(self, user, place, value):
        """Sets user's knowledge of the given place.

        :param user: ID of the user.
        :type user: int
        :param place: ID of the place.
        :type place: int
        :param value: Value of the estimated knowledge.
        :type value: float
        """
        self.knowledge.set_value((user, place), 'knowledge', value)

    def get_practices(self, user, place):
        """Returns list of previous responses of the given user on
        the given place.

        :param user: ID of the user.
        :type user: int
        :param place: ID of the place.
        :type place: int
        """
        now = datetime.now()
        practices = self.practices[user, place]
        return [(now - t).total_seconds() for t in practices]

    def apply(self, func, axis=1):
        """Applies function along input axis of DataFrame.

        :param func: Function to apply to each column or row.
        :type func: callable
        :param axis: `0` if the function should be applied to each column,
            `1` if the function should be applied to each row.
        :type axis: int
        """
        self.data.apply(func, axis=axis)

    def user_asked(self, user):
        """Returns number of questions the given user was asked.

        :param user: ID of the user.
        :type user: int
        """
        return len(self.data[self.data['user'] == user])

    def place_asked(self, place):
        """Returns how many times the given place has been asked.

        :param place: ID of the place.
        :type place: int
        """
        return len(self.data[self.data['place_asked'] == place])

    def to_datetime(self, date_str):
        """Deserializes given datetime.

        :param date_str: DateTime given as string.
        :type date_str: str
        """
        return datetime.strptime(date_str, self.datetime_format)


class Model(object):
    """Abstract model class."""

    def __init__(self):
        self.data_set = None

    def respect_guess(self, prediction, options):
        """Updates prediction with respect to guessing paramter.

        :param prediction: Prediction calculated so far.
        :type prediction: float
        :param options: Number of options for the question.
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

    def update(self, question):
        """Performes update of the :class:`DataSet` instance based on
        the estimated knowledge, skill or difficulty.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        raise NotImplementedError()

    def train(self, data_set):
        """Trains the model on given data set.

        :param data_set: Data set on which to train the model.
        :type data_set: :class:`DataSet`
        """
        raise NotImplementedError()


class EloModel(Model):
    """Predicts correctness of answers using Elo Rating System."""

    def __init__(self, alpha=1, beta=0.05):
        super(EloModel, self).__init__()

        self.alpha = alpha
        self.beta = beta

    def uncertanty(self, n):
        return self.alpha / (1 + self.beta * n)

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        skill = self.data_set.get_skill(question.user)
        difficulty = self.data_set.get_difficulty(question.place_asked)

        prediction = tools.sigmoid(skill - difficulty)
        return self.respect_guess(prediction, question.number_of_options)

    def update(self, question):
        """Performes update of the :class:`DataSet` instance based on
        the estimated knowledge, skill or difficulty.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        shift = question.correct - self.predict(question)

        q_u = self.uncertanty(self.data_set.user_asked(question.user))
        q_d = self.uncertanty(self.data_set.place_asked(question.place_asked))

        skill = self.data_set.get_skill(question.user)
        diffi = self.data_set.get_difficulty(question.place_asked)

        self.set_skill(question.user, skill + q_u * shift)
        self.set_difficulty(question.place_asked, diffi + q_d * shift)

    def train(self, data_set):
        """Trains the model on given data set.

        :param data_set: Data set on which to train the model.
        :type data_set: :class:`DataSet`
        """
        self.data_set = data_set
        self.data_set.apply(self.update)


class PFAModel(Model):
    """PFA model for estimation of current knowledge."""

    def __init__(self, gamma=3.4, delta=0.3):
        super(PFAModel, self).__init__()

        self.gamma = gamma
        self.delta = delta

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        args = [question.user, question.place_asked]
        knowledge = self.data_set.get_knowledge(*args)

        prediction = tools.sigmoid(knowledge)
        return self.respect_guess(prediction, question.number_of_options)

    def update(self, question):
        """Performes update of the :class:`DataSet` instance based on
        the estimated knowledge, skill or difficulty.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        args = [question.user, question.place_asked]
        knowledge = self.data_set.get_knowledge(*args)

        prediction = self.predict(question)
        result = knowledge + self.gamma * (question.correct - prediction)

        self.data_set.set_knowledge(*(args + [result]))

    def train(self, data_set):
        """Trains the model on given data set.

        :param data_set: Data set on which to train the model.
        :type data_set: :class:`DataSet`
        """
        self.data_set = data_set
        self.data_set.apply(self.update)


class PFAWithSpacing(PFAModel):
    """Extended version of PFA that takes into account the effect of
    forgetting and spacing.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 4)
        kwargs.setdefault('delta', 1)

        self.spacing_rate = kwargs.pop('spacing_rate', 0)
        self.decay_rate = kwargs.pop('decay_rate', 0.2)

        super(PFAWithSpacing, self).__init__(*args, **kwargs)

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        args = [question.user, question.place_asked]

        practices = self.data_set.get_practices(*args)
        knowledge = self.data_set.get_knowledge(*args)

        strength = tools.memory_strength(
            practices,
            spacing_rate=self.spacing_rate,
            decay_rate=self.decay_rate,
        )

        prediction = tools.sigmoid(knowledge + strength)
        return self.respect_guess(prediction, question.number_of_options)
