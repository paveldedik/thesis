# -*- coding: utf-8 -*-


import pandas as pd
from datetime import datetime

import tools


class Model(object):
    """Abstract model class."""

    datetime_format = '%Y-%m-%d %H:%M:%S'

    def __init__(self, data):
        self.data = data

    @tools.cached_property
    def places(self):
        """Returns all places in the given data set."""
        return self.data['place_asked'].unique()

    @tools.cached_property
    def users(self):
        """Returns all users in the given data set."""
        return self.data['user'].unique()

    def to_datetime(self, date_str):
        """Deserializes given datetime."""
        return datetime.strptime(date_str, self.datetime_format)

    def respect_guess(self, prediction, options):
        """Updates prediction with respect to guessing paramter."""
        if options > 0:
            val = 1 / options
            return val + (1 - val) * prediction
        else:
            return prediction

    def predict(self, row):
        """Returns probability of correct answer for given row."""
        raise NotImplementedError()

    def train(self):
        """Trains model on given data set."""
        raise NotImplementedError()


class EloModel(Model):
    """Predicts correctness of answers using ELO model."""

    def __init__(self, data, alpha=1, beta=0.05):
        super(EloModel, self).__init__(data)

        self.alpha = alpha
        self.beta = beta

        self.data = tools.get_prior(self.data)
        self.data = tools.prepare_data(self.data)

        self.difficulties = pd.DataFrame(
            columns=['difficulty'],
            index=pd.Index([], name='place_asked')
        )
        self.skills = pd.DataFrame(
            columns=['skill'],
            index=pd.Index([], name='user')
        )

    @tools.cached_property
    def merged(self):
        """Contains data with calculated difficulties and skills."""
        return pd.merge(
            pd.merge(self.data, self.difficulties.reset_index(),
                     on='place_asked'),
            self.skills.reset_index(), on='user'
        )

    def get_skill(self, user):
        """Returns estimated skill of given user."""
        if user in self.skills.index:
            return self.skills.get_value(user, 'skill')
        else:
            return 0

    def get_difficulty(self, place):
        """Returns estimated difficulty of given place."""
        if place in self.difficulties.index:
            return self.difficulties.get_value(place, 'difficulty')
        else:
            return 0

    def predict(self, row):
        """Predicts correctness of an answer."""
        user, place, opts = row[['user', 'place_asked', 'number_of_options']]
        diff = self.get_skill(user) - self.get_difficulty(place)
        prediction = tools.sigmoid(diff)
        return self.respect_guess(prediction, opts)

    def update(self, row):
        """Updates skills and difficulties."""
        def fun(n, alpha=self.alpha, beta=self.beta):
            return alpha / (1 + beta * n)

        user, place, correct = row[['user', 'place_asked', 'correct']]
        shift = correct - self.predict(row)

        ans_u = len(self.data[self.data['user'] == user])
        ans_d = len(self.data[self.data['place_asked'] == place])

        skill = self.get_skill(user) + fun(ans_u) * shift
        difficulty = self.get_difficulty(place) - fun(ans_d) * shift

        self.skills.set_value(user, 'skill', skill)
        self.difficulties.set_value(place, 'difficulty', difficulty)

    def train(self):
        """Trains the model on given data set."""
        self.data.apply(self.update, axis=1)


class PFAModel(Model):
    """PFA model for estimation of current knowledge."""

    def __init__(self, data, prior_model, gamma=3.4, delta=0.3):
        super(PFAModel, self).__init__(data)
        self.data = tools.prepare_data(data)

        self.gamma = gamma
        self.delta = delta

        self.prior = prior_model
        self.knowledge = pd.DataFrame(
            columns=['knowledge'],
            index=pd.MultiIndex([[], []], [[], []], names=['user', 'place'])
        )

    def get_knowledge(self, user, place):
        """Returns user's knowledge of a given place."""
        if (user, place) in self.knowledge.index:
            return self.knowledge.get_value((user, place), 'knowledge')
        else:
            skill = self.prior.get_skill(user)
            difficulty = self.prior.get_difficulty(place)
            return skill - difficulty

    def predict(self, row):
        """Predicts probability of correct answer."""
        user, place, opts = row[['user', 'place_asked', 'number_of_options']]
        know = self.get_knowledge(user, place)
        prediction = tools.sigmoid(know)
        return self.respect_guess(prediction, opts)

    def update(self, row):
        """Updates user's knowledge of a place."""
        user, place, correct = row[['user', 'place_asked', 'correct']]

        knowledge = self.get_knowledge(user, place)
        prediction = self.predict(row)

        self.data.loc[row.name, 'prediction'] = prediction

        if correct:
            result = knowledge + self.gamma * (1 - prediction)
        else:
            result = knowledge + self.delta * (0 - prediction)

        self.knowledge.set_value((user, place), 'knowledge', result)

    def train(self):
        """Trains the model on given data set."""
        sorted_data = self.data.sort(['inserted'])
        sorted_data.apply(self.update, axis=1)


class PFAWithSpacing(PFAModel):
    """Extended version of PFA."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('gamma', 4)
        kwargs.setdefault('delta', 1)

        self.times = {}
        self.spacing_rate = kwargs.pop('spacing_rate', 0.2)
        self.decay_rate = kwargs.pop('decay_rate', 0.05)

        super(PFAWithSpacing, self).__init__(*args, **kwargs)

    def get_times(self, user, place, inserted):
        """Returns list of previous responses of the given user on
        the given place.
        """
        inserted = self.to_datetime(inserted)
        all_times = self.responses[user, place]
        return [(inserted - t).total_seconds() for t in all_times]

    def predict(self, row):
        """Predicts probability of correct answer with respect to
        spacing and forgeting effect.
        """
        user, place, inserted = row[['user', 'place_asked', 'inserted']]

        times = self.get_times(user, place, inserted)
        knowledge = self.get_knowledge(user, place)
        difficulty = self.prior.get_difficulty(place)

        strength = tools.memory_strength(
            times,
            spacing_rate=self.spacing_rate,
            decay_rate=self.decay_rate * difficulty,
        )
        return tools.sigmoid(knowledge + strength)

    def update(self, row):
        """Updates user's knowledge of a place."""
        user, place, correct = row[['user', 'place_asked', 'correct']]

        knowledge = self.get_knowledge(user, place)
        prediction = self.predict(row)

        self.data.loc[row.name, 'prediction'] = prediction

        if correct:
            result = knowledge + self.gamma * (1 - prediction)
        else:
            result = knowledge + self.delta * (1 - prediction)

        self.knowledge.set_value((user, place), 'knowledge', result)

    def train(self):
        """Trains the model on given data set."""
        for index, group in self.data.groupby(['user', 'place_asked']):
            times = sorted(self.to_datetime(time) for time in group)
            self.times[index] = times
        super(PFAWithSpacing, self).train()
