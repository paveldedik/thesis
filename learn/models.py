# -*- coding: utf-8 -*-


import pandas as pd

import tools


class Model(object):
    """Abstract model class."""

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

    def predict(self, row):
        """Returns probability of correct answer for given row."""
        raise NotImplementedError()

    def train(self):
        """Trains model on given data set."""
        raise NotImplementedError()


class EloModel(Model):
    """Predicts correctness of answers using ELO model."""

    def __init__(self, data,
                 difficulties=None, skills=None,
                 alpha=1, beta=0.05):
        super(EloModel, self).__init__(data)

        self.alpha = alpha
        self.beta = beta

        self.data = tools.get_prior(self.data)
        self.data = tools.prepare_data(self.data)

        self.difficulties = difficulties or pd.DataFrame(
            columns=['difficulty'],
            index=pd.Index([], name='place_asked')
        )
        self.skills = skills or pd.DataFrame(
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
        probability = tools.sigmoid(diff)

        if opts > 0:
            val = 1 / opts
            return val + (1 - val) * probability
        else:
            return probability

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
        self.data['prediction'] = self.data.apply(self.predict, axis=1)


class PfaModel(Model):
    """PFA model for estimation of current knowledge."""

    def __init__(self, data, prior_model,
                 knowledge=None, gamma=3.4, delta=0.3):
        super(PfaModel, self).__init__(data)
        self.data = tools.prepare_data(data)

        self.gamma = gamma
        self.delta = delta

        self.prior = prior_model
        self.knowledge = knowledge or pd.DataFrame(
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
        probability = tools.sigmoid(know)
        if opts > 0:
            val = 1 / opts
            return val + (1 - val) * probability
        else:
            return probability

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
        groups = self.data.sort(
            ['user', 'place_asked', 'inserted']
        ).groupby(['user', 'place_asked'])

        for index_value, group in groups:
            group.apply(self.update, axis=1)
