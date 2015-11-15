# -*- coding: utf-8 -*-

"""
Evaluation Models
=================

"""

from __future__ import division

from copy import copy
from collections import defaultdict

import numpy as np
import pandas as pd

import tools


__all__ = (
    'DummyPriorModel',
    'EloModel',
    'EloResponseTime',
    'PFAModel',
    'PFAExt',
    'PFAExtTiming',
    'PFAExtStaircase',
    'PFAExtSpacing',
    'PFAGong',
    'PFAGongTiming',
)


class Question(object):
    """Representation of a question."""

    def __init__(self, **kwargs):
        self.id = kwargs.pop('id')
        self.user_id = kwargs.pop('user_id')
        self.place_id = kwargs.pop('place_id')
        self.type = kwargs.pop('type')
        self.inserted = kwargs.pop('inserted')
        self.options = kwargs.pop('options')


class Answer(Question):
    """Answer to a question."""

    def __init__(self, **kwargs):
        super(Answer, self).__init__(**kwargs)

        self.place_answered = kwargs.pop('place_answered')
        self.response_time = kwargs.pop('response_time')
        self.is_correct = kwargs.pop('is_correct')


class User(object):
    """Returns a user with given ID.

    :param user_id: ID of the user.
    :type user_id: int
    """
    def __init__(self, user_id):
        self.id = user_id
        self.skill_increments = []

    @property
    def skill(self):
        """Skill of the user."""
        return sum(self.skill_increments)

    @property
    def answers_count(self):
        """Number of answer of the user (equal to the number of
        skill increments.
        """
        return len(self.skill_increments)

    def inc_skill(self, increment):
        """Increments the skill of the user.

        :param increment: Increment (or decrement) of the skill.
        :type increment: int
        """
        self.skill_increments += [increment]


class Place(object):
    """Returns a place with given ID.

    :param place_id: ID of the place.
    :type place_id: int
    """
    def __init__(self, place_id):
        self.id = place_id
        self.difficulty_increments = []

    @property
    def difficulty(self):
        """Difficulty of the place."""
        return sum(self.difficulty_increments)

    @property
    def answers_count(self):
        """Number of answer for the place (equal to the number of
        difficulty increments.
        """
        return len(self.difficulty_increments)

    def inc_difficulty(self, increment):
        """Increments the difficulty of the place.

        :param increment: Increment (or decrement) of the difficulty.
        :type increment: int
        """
        self.difficulty_increments += [increment]


class Item(object):
    """Item representation.

    :param prior: Prior skills of users and difficulties of places.
    :type prior: dictionary
    :param user_id: ID of the user.
    :type user_id: int
    :param place_id: ID of the place.
    :type place_id: int
    """

    def __init__(self, prior, user_id, place_id):
        self.prior = prior
        self.user_id = user_id
        self.place_id = place_id

        self.practices = []
        self.knowledge_increments = []

    @property
    def user(self):
        """User answering the item."""
        return self.prior.users[self.user_id]

    @property
    def place(self):
        """Place of the item being asked."""
        return self.prior.places[self.place_id]

    @property
    def knowledge(self):
        """Knowledge of the item by the user."""
        return (
            (self.user.skill - self.place.difficulty)
            + sum(self.knowledge_increments)
        )

    @property
    def correct(self):
        """List of correct answers."""
        return [ans for ans in self.practices if ans.is_correct]

    @property
    def incorrect(self):
        """List of incorrect answers."""
        return [ans for ans in self.practices if not ans.is_correct]

    @property
    def last_inserted(self):
        """Returns the time of the last answer for this item
        or :obj:`None` if the item was never answered before.
        """
        if self.practices:
            return self.practices[-1].inserted

    @property
    def any_incorrect(self):
        """:obj:`True` if at least one of the practiced item
        was answered incorrectly, otherwise :obj:`False`.
        """
        return any(not answer.is_correct for answer in self.practices)

    def get_diffs(self, current):
        """Returns list of previous practices expresed as the number
        of seconds that passed between *current* practice and all
        the *previous* practices.

        :param current: Datetime of the current practice.
        :type place: string
        """
        return [
            tools.time_diff(current, prior.inserted)
            for prior in self.practices
        ]

    def inc_knowledge(self, increment):
        """Increments the knowledge of the user of the item.

        :param increment: Increment (or decrement) of the knowledge.
        :type increment: int
        """
        self.knowledge_increments += [increment]

    def add_practice(self, answer):
        """Registers new practice of the item.

        :param answer: Information about the answer.
        :type answer: :class:`pandas.Series` or :class:`Answer`
        """
        if isinstance(answer, pd.Series):
            self.practices += [Answer(**answer.to_dict())]
        else:
            self.practices += [copy(answer)]


class Model(object):
    """Abstract model class."""

    ABBR = None

    def respect_guess(self, prediction, options):
        """Updates prediction with respect to guessing paramter.

        :param prediction: Prediction calculated so far.
        :type prediction: float
        :param options: Number of options in the multiple-choice question.
        :type options: int
        """
        if options:
            val = 1 / len(options)
            return val + (1 - val) * prediction
        else:
            return prediction

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        raise NotImplementedError()

    def update(self, answer):
        """Performes an update of skills, difficulties or knowledge.

        :param answer: Asked question.
        :type answer: :class:`pandas.Series` or :class:`Answer`
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


class DummyPriorModel(Model):
    """Dummy model that sets all skills of users and difficulties
    of places to zero.
    """

    class _User(object):
        """Returns a user with given ID."""

        def __init__(self, skill):
            self.skill = skill

    class _Place(object):
        """Returns a place with given ID."""

        def __init__(self, difficulty):
            self.difficulty = difficulty

    def __init__(self, skill=0.0, difficulty=0.0):
        self.users = defaultdict(lambda: self._User(skill))
        self.places = defaultdict(lambda: self._Place(difficulty))

    def update(self, answer):
        pass

    def train(self, data):
        pass


class EloModel(Model):
    """Predicts correctness of answers using Elo Rating System.
    The model is parametrized with `alpha` and `beta`. These parameters
    affect the uncertainty function.
    """
    ABBR = 'Elo'

    def __init__(self, alpha=1, beta=0.05):
        self.alpha = alpha
        self.beta = beta

        self.init_model()

    def init_model(self):
        """Initializes two attributes of the model. Both attributes are
        dataframes. The first attribute represents difficulties of countries.
        The second attribute represents global knowledge of students.
        """
        self.places = tools.keydefaultdict(Place)
        self.users = tools.keydefaultdict(User)

        self.predictions = {}

    def uncertainty(self, n):
        """Uncertainty function. The purpose is to make each update on
        the model trained with sequence of `n` answers less and less
        significant as the number of prior answers is bigger.

        :param n: Number of user's answers or total answers to a place.
        :type n: int
        """
        return self.alpha / (1 + self.beta * n)

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        user = self.users[question.user_id]
        place = self.places[question.place_id]

        prediction = tools.sigmoid(user.skill - place.difficulty)
        return self.respect_guess(prediction, question.options)

    def update(self, answer):
        """Updates skills of users and difficulties of places according
        to given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series`
        """
        user = self.users[answer.user_id]
        place = self.places[answer.place_id]

        prediction = self.predict(answer)
        shift = answer.is_correct - prediction

        user.inc_skill(self.uncertainty(user.answers_count) * shift)
        place.inc_difficulty(-(self.uncertainty(place.answers_count) * shift))

        self.predictions[answer.id] = prediction

    def train(self, data):
        """Trains the model on given data set.

        :param data: Data set on which to train the model.
        :type data: :class:`pandas.DataFrame`
        """
        self.init_model()
        data = tools.first_answers(data)
        data.sort(['inserted']).apply(self.update, axis=1)

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
    ABBR = 'Elo/RT'

    def __init__(self, *args, **kwargs):
        self.zeta = kwargs.pop('zeta', 3)

        super(EloResponseTime, self).__init__(*args, **kwargs)

    def update(self, answer):
        """Updates skills of users and difficulties of places according
        to given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series` or :class:`Answer`
        """
        user = self.users[answer.user_id]
        place = self.places[answer.place_id]

        prediction = self.predict(answer)
        level = tools.automaticity_level(answer.response_time)

        prob = (prediction * self.zeta + level) / (self.zeta + 1)
        shift = answer.is_correct - prob

        user.inc_skill(self.uncertainty(user.answers_count) * shift)
        place.inc_difficulty(-(self.uncertainty(place.answers_count) * shift))

        self.predictions[answer.id] = prediction


class PFAModel(Model):
    """Standard Performance Factor Analysis.

    :param gamma: The significance of the update when the student
        answered correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answered incorrectly.
    :type delta: float
    """
    ABBR = 'PFA'

    def __init__(self, prior=None, gamma=3.4, delta=-0.3):
        super(PFAModel, self).__init__()

        self.prior = prior or DummyPriorModel()
        self.gamma = gamma
        self.delta = delta

        self.init_model()

    def init_model(self):
        """Initializes attribute of the model that stores current
        knowledge of places for all students.
        """
        self.items = tools.keydefaultdict(
            lambda *args: Item(self.prior, *args)
        )
        self.predictions = {}

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        item = self.items[question.user_id, question.place_id]

        knowledge = (
            item.knowledge +
            self.gamma * len(item.correct) +
            self.delta * len(item.incorrect)
        )

        return tools.sigmoid(knowledge)

    def update(self, answer):
        """Performes update of current knowledge of a user based on the
        given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series` or :class:`Answer`
        """
        item = self.items[answer.user_id, answer.place_id]

        if not item.practices:
            self.prior.update(answer)

        prediction = self.predict(answer)
        self.predictions[answer.id] = prediction

        item.add_practice(answer)

    def train(self, data):
        """Trains the model on given data set.

        :param data: Data set on which to train the model.
        :type data: :class:`pandas.DataFrame`
        """
        self.init_model()
        data.sort(['inserted']).apply(self.update, axis=1)

    @classmethod
    def split_data(self, data):
        """Classmethod that splits data into training set and test set.

        :param data: The object containing data.
        :type data: :class:`pandas.DataFrame`.
        """
        test_set = tools.last_answers(data)
        train_set = data[~data['id'].isin(test_set['id'])]

        return train_set, test_set


class PFAExt(PFAModel):
    """PFA model for estimation of current knowledge.

    :param gamma: The significance of the update when the student
        answered correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answered incorrectly.
    :type delta: float
    """
    ABBR = 'PFA/E'

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        item = self.items[question.user_id, question.place_id]
        prediction = tools.sigmoid(item.knowledge)
        return self.respect_guess(prediction, question.options)

    def update(self, answer):
        """Performes update of current knowledge of a user based on the
        given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series` or :class:`Answer`
        """
        item = self.items[answer.user_id, answer.place_id]

        if not item.practices:
            self.prior.update(answer)

        prediction = self.predict(answer)
        self.predictions[answer.id] = prediction

        item.add_practice(answer)

        if answer.is_correct:
            item.inc_knowledge(self.gamma * (1 - prediction))
        else:
            item.inc_knowledge(self.delta * prediction)


class PFAExtTiming(PFAExt):
    """Alternative version of :class:`PFAExtSpacing` which ignores
    spacing effect. Only forgetting is considered.

    :param gamma: The significance of the update when the student
        answered correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answered incorrectly.
    :type delta: float
    :param time_effect_fun: Time effect function.
    :type time_effect_fun: callable
    """
    ABBR = 'PFA/E/T'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 2.3)
        kwargs.setdefault('delta', -0.9)

        time_effect = lambda t: 80 / t
        self.time_effect = kwargs.pop('time_effect_fun', time_effect)

        super(PFAExtTiming, self).__init__(*args, **kwargs)

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        item = self.items[question.user_id, question.place_id]

        if item.practices:
            seconds = tools.time_diff(question.inserted, item.last_inserted)
            time_effect = self.time_effect(seconds)
        else:
            time_effect = 0

        prediction = tools.sigmoid(item.knowledge + time_effect)
        return self.respect_guess(prediction, question.options)


class PFAExtStaircase(PFAExtTiming):
    """Alternative version of :class:`PFAESpacing` which ignores
    spacing effect. Only forgetting is considered given by staircase
    fucntion.

    :param gamma: The significance of the update when the student
        answered correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answered incorrectly.
    :type delta: float
    :param time_effect_fun: Values for staircase function.
    :type time_effect_fun: dict (tuples as keys)
    """
    ABBR = 'PFA/E/T staircase'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 2.5)
        kwargs.setdefault('delta', -0.8)

        self.staircase = tools.intervaldict(kwargs.pop('staircase'))
        self.time_effect = lambda k: self.staircase[k]

        super(PFAExtTiming, self).__init__(*args, **kwargs)


class PFAExtSpacing(PFAExtTiming):
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
    ABBR = 'PFA/E/S'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 2.8)
        kwargs.setdefault('delta', -0.7)

        self.spacing_rate = kwargs.pop('spacing_rate', 0)
        self.decay_rate = kwargs.pop('decay_rate', 0.18)
        self.iota = kwargs.pop('iota', 1.5)

        super(PFAExtSpacing, self).__init__(*args, **kwargs)

    def memory_strength(self, question):
        """Estimates memory strength of an item.

        :param question: Asked question.
        :type question: :class:`pandas.Series`
        """
        item = self.items[question.user_id, question.place_id]
        practices = item.get_diffs(question.inserted)

        if len(practices) > 0:
            return self.iota + tools.memory_strength(
                filter(lambda x: x > 0, practices),
                spacing_rate=self.spacing_rate,
                decay_rate=self.decay_rate,
            )

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        item = self.items[question.user_id, question.place_id]

        if item.any_incorrect:
            strength = self.memory_strength(question)
        else:
            strength = 0

        prediction = tools.sigmoid(item.knowledge + strength)
        return self.respect_guess(prediction, question.options)


class PFAGong(PFAModel):
    """Yue Gong's extended Performance Factor Analysis.

    :param gamma: The significance of the update when the student
        answers correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answers incorrectly.
    :type delta: float
    :param decay: Decay rate of answers.
    :type decay: float
    """
    ABBR = 'PFA/G'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 2.1)
        kwargs.setdefault('delta', -0.8)

        self.decay = kwargs.pop('decay', 0.8)

        super(PFAGong, self).__init__(*args, **kwargs)

    def get_weights(self, item, question):
        """Returns weights of previous answers to the given item.

        :param item: *Item* (i.e. practiced place by a user).
        :type item: :class:`Item`
        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        correct_weights = [
            ans.is_correct * self.decay ** t for t, ans
            in tools.reverse_enumerate(item.practices)
        ]
        incorrect_weights = [
            (1 - ans.is_correct) * self.decay ** t for t, ans
            in tools.reverse_enumerate(item.practices)
        ]
        return sum(correct_weights), sum(incorrect_weights)

    def predict(self, question):
        """Returns probability of correct answer for given question.

        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        item = self.items[question.user_id, question.place_id]
        correct_weight, incorrect_weight = self.get_weights(item, question)

        knowledge = (
            item.knowledge +
            self.gamma * correct_weight +
            self.delta * incorrect_weight
        )

        prediction = tools.sigmoid(knowledge)
        return self.respect_guess(prediction, question.options)

    def update(self, answer):
        """Performes update of current knowledge of a user based on the
        given answer.

        :param answer: Answer to a question.
        :type answer: :class:`pandas.Series` or :class:`Answer`
        """
        item = self.items[answer.user_id, answer.place_id]

        if not item.practices:
            self.prior.update(answer)

        prediction = self.predict(answer)
        self.predictions[answer.id] = prediction

        item.add_practice(answer)


class PFAGongTiming(PFAGong):
    """Performance Factor Analysis combining some aspects of both
    the Yue Gong's PFA and the ACT-R model.

    :param gamma: The significance of the update when the student
        answers correctly.
    :type gamma: float
    :param delta: The significance of the update when the student
        answers incorrectly.
    :type delta: float
    :param time_effect_fun: Time effect function.
    :type time_effect_fun: callable
    """
    ABBR = 'PFA/G/T'

    def __init__(self, *args, **kwargs):
        time_effect = lambda t: 1.1 - 0.08 * np.log(t)
        self.time_effect = kwargs.pop('time_effect_fun', time_effect)

        kwargs.setdefault('gamma', 1.7)
        kwargs.setdefault('delta', 0.5)

        super(PFAGong, self).__init__(*args, **kwargs)

    def get_weights(self, item, question):
        """Returns weights of previous answers to the given item.

        :param item: *Item* (i.e. practiced place by a user).
        :type item: :class:`Item`
        :param question: Asked question.
        :type question: :class:`pandas.Series` or :class:`Question`
        """
        correct_weights = [
            max(ans.is_correct * self.time_effect(diff), 0) for ans, diff
            in zip(item.practices, item.get_diffs(question.inserted))
        ]
        incorrect_weights = [
            (1 - ans.is_correct) * self.time_effect(diff) for ans, diff
            in zip(item.practices, item.get_diffs(question.inserted))
        ]
        return sum(correct_weights), sum(incorrect_weights)
