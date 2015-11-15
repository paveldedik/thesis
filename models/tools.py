# -*- coding: utf-8 -*-

"""
Miscellaneous Helpers and Utils
===============================

"""

from __future__ import division

import sys
import csv
import ast
from datetime import datetime
from collections import defaultdict

import pytz
import numpy as np
import pandas as pd
from scipy import optimize
from IPython.display import clear_output
from sklearn.metrics import mean_squared_error

from . import config


def echo(msg, clear=True):
    """Prints the passed message. If the parameter `clear` is set to
    :obj:`True`, IPython's console output is cleared beforhand.

    :param msg: Message to write to standard output.
    :type msg: string
    :param clear: Clear the output beforhand (works only for IPython'
        console output).
    :type clear: bool
    """
    if clear:
        clear_output()
    print msg
    sys.stdout.flush()


def load_data(path=config.DATA_ANSWERS_PATH,
              users_path=config.DATA_USERS_PATH,
              limit=10000, offset=0, echo_loaded=True):
    """Loads CSV file with answers into :class:`pandas.DataFrame`.

    :param path: Path to the CSV file with the answers of users.
    :type path: str
    :param limit: Number of loaded rows at most.
    :type limit: int
    :param offset: Number of rows to skip.
    :type offset: int
    :param users_path: Path to the CSV file containing all users.
        This is necessary so that first answers of users are truly first.
        The CSV file is used to filter the answers correctly.
    :type users_path: string
    """
    data = pd.read_csv(
        # Skip at least the first column, which is the header containing
        # the names of columns. This is usually not necessary, but we pass
        # the list of the columns' aliases explicitly here (names=...).
        path, skiprows=int(offset or 1), nrows=int(limit), sep=';',
        names=config.ANSWERS_COLUMNS)
    data = prepare_data(data)

    if users_path is not None:
        users = pd.read_csv(users_path)
        new_users = users[users['first_answer_id'] >= min(data['id'])]
        data = data[data['user_id'].isin(new_users['user_id'])]

    if echo_loaded:
        echo('Loaded {} answers.'.format(len(data)), clear=False)
    return data


def load_places(path=config.DATA_PLACES_PATH, index_col='id'):
    """Loads CSV file with places into :class:`pandas.DataFrame`.

    :param path: Path to CSV file.
    :type path: str
    """
    return pd.read_csv(path, index_col=index_col, sep=';')


def load_place_types(path=config.DATA_PLACE_TYPES_PATH, index_col='name'):
    """Loads CSV file with places into :class:`pandas.DataFrame`.

    :param path: Path to CSV file.
    :type path: str
    """
    return pd.read_csv(path, index_col=index_col, sep=';')


def generate_users(data, users_path=config.DATA_USERS_PATH):
    """Exports users from given data into CSV file.

    :param data: Data to export users from.
    :type data: :class:`pandas.DataFram`
    :param users_path: Where to save the exported users.
    :type users_path: string
    """
    users = {}
    for index, row in data.sort(['inserted']).iterrows():
        if row.user_id not in users:
            users[row.user_id] = (row.id, row.inserted)

    with open(users_path, 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(config.USERS_COLUMNS)
        for user_id, (answer_id, answer_inserted) in users.items():
            csv_out.writerow((user_id, answer_id, answer_inserted))


def prepare_data(data):
    """Prepares data for models (e.g. adds *is_correct* column).

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    data['is_correct'] = (data['place_id'] ==
                          data['place_answered']).astype(int)
    data['options'] = data['options'].apply(ast.literal_eval)
    data['inserted'] = data['inserted'].apply(to_datetime)
    return data[[column for column in data.columns
                 if column not in config.IGNORED_COLUMNS]].copy()


def first_answers(data):
    """Modifies the given data object so that only the first answers of
    students are contained in the data.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    data = data.sort(
        ['user_id', 'place_id', 'inserted']
    ).groupby(
        ['user_id', 'place_id']
    ).first()
    return data.reset_index()


def last_answers(data):
    """Modifies the given data object so that only the last answers of
    students are contained in the data.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    data = data.sort(
        ['user_id', 'place_id', 'inserted']
    ).groupby(
        ['user_id', 'place_id']
    ).last()
    return data.reset_index()


def unknown_answers(data):
    """Modifies the given data set so that only the answers the user
    didn't answer correctly at first are contained in the data.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    def tuplify_user_place(serie):
        return (serie['user_id'], serie['place_id'])

    first = first_answers(data)
    unknowns = first[first['is_correct'] == 0]

    all_user_place = data.apply(tuplify_user_place, axis=1)
    unk_user_place = unknowns.apply(tuplify_user_place, axis=1)

    mask = all_user_place.isin(unk_user_place)
    return data[mask].reset_index(drop=True)


def add_spacing(data):
    """Appends the given data set with spacing information of earch item.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    answers = {}
    data['spacing'] = np.nan

    def set_spacing(row):
        index = (row.user_id, row.place_id)
        if index in answers:
            data.loc[row.name, 'spacing'] = \
                time_diff(row.inserted, answers[index])
        answers[index] = row.inserted

    data.sort(['inserted']).apply(set_spacing, axis=1)
    return data


def split_data(data, ratio=0.7):
    """Splits data into test set and training set.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    :param ratio: What portion of data to include in the training set
        and the test set. :obj:`0.5` means that the data will be
        distributed equaly.
    :type ratio: float
    """
    threshold = int(len(data) * ratio)
    return data[:threshold], data[threshold:]


def shuffle_data(data):
    """Shuffles the rows in the data set.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    items = np.random.permutation(data.index)
    return data.reindex(items)


def random_data(data, limit=1000):
    """Randomly selects rows from given data set.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    items = np.random.permutation(data.index)[:limit]
    return data.ix[items]


def add_ones(data):
    """Appends the data set with a column containing *ones*.

    :param data: The object containing data.
    :type data: :class:`pandas.DataFrame`.
    """
    columns = list(data.columns)
    data['ones'] = np.ones(len(data))
    data = data[['ones'] + columns]
    return data


def rmse(y_true, y_pred):
    """Calculates Root Mean Squered Error for given vectors.

    :param y_true: Vector containing *true* values.
    :type y_true: list, :class:`numpy.array` or :class:`pandas.Series`
    :param y_pred: Vector containing *predicted* values.
    :type y_true: list, :class:`numpy.array` or :class:`pandas.Series`
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def log_likelihood(y_true, y_pred):
    """Calculates Log Likelihood for given vectors.

    :param y_true: Vector containing *true* values.
    :type y_true: list, :class:`numpy.array` or :class:`pandas.Series`
    :param y_pred: Vector containing *predicted* values.
    :type y_true: list, :class:`numpy.array` or :class:`pandas.Series`
    """
    return (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).sum()


def sigmoid(x):
    """Implementation of the sigmoid (logistic) function.

    :param x: Function parameter.
    :type x: int or float
    """
    return 1 / (1 + np.exp(-x))


def cost(theta, data, y):
    """Vectorized implementation of the cost function for logistic
    regression.

    :param theta: Vector with parameters of the hyphotesis function.
    :type theta: list, :class:`numpy.array` or :class:`pandas.Series`
    :param data: Data object containing a set of training examples.
    :type data: :class:`pandas.DataFrame` or :class:`numpy.matrix`
    :param y: The array containing classifications of training examples.
    :type y: list, :class:`numpy.array` or :class:`pandas.Series`
    """
    hyph = sigmoid(np.dot(data, theta))
    calc = (-1 * y) * np.log(hyph) - (1 - y) * np.log(1 - hyph)
    return sum(calc) / len(y)


def grad(theta, data, y):
    """Vectorized implementation of the algorithm that evaluates
    gradient for given parameters.

    :param theta: Vector with parameters of the hyphotesis function.
    :type theta: list, :class:`numpy.array` or :class:`pandas.Series`
    :param data: Data object containing a set of training examples.
    :type data: :class:`pandas.DataFrame` or :class:`numpy.matrix`
    :param y: The array containing classifications of training examples.
    :type y: list, :class:`numpy.array` or :class:`pandas.Series`
    """
    hyph = sigmoid(np.dot(data, theta))
    return np.dot(hyph - y, data) / y.size


def find_theta(theta, data, y, maxiter=1000):
    """Runs BFGS algoritm that minimizes cost function and returns
    minimized theta.

    :param theta: Vector with parameters of the hyphotesis function.
    :type theta: list, :class:`numpy.array` or :class:`pandas.Series`
    :param data: Data object containing a set of training examples.
    :type data: :class:`pandas.DataFrame` or :class:`numpy.matrix`
    :param y: The array containing classifications of training examples.
    :type y: list, :class:`numpy.array` or :class:`pandas.Series`
    :param maxiter: Maximum number of iterations to perform.
    :type maxiter: int
    """
    return optimize.fmin_bfgs(
        cost, theta, fprime=grad, args=(data, y), maxiter=maxiter
    )


def memory_strength(times, spacing_rate=0.2, decay_rate=0.18):
    """Calculates memory strength based on given vector of practices.
    Each element *t_i* in the vector indicates how long ago the *i*-th
    practice occured.

    :param times: Vector containing all prior practices.
        Each practice is represented as the number of seconds that
        passed since the student answered the question.
    :type times: list, :class:`numpy.array` or :class:`pandas.Series`
    :param spacing_rate: The significance of the spacing effect. Lower
        values make the effect less significant. If the spacing rate
        is set to zero, the model is unaware of the spacing effect.
    :type spacing_rate: float
    :param decay_rate: The significance of the forgetting effect. Higher
        values of decay rate make the students forget the item faster
        and vice versa.
    :type decay_rate: float
    """
    strengths = [-np.inf]
    get_decay = lambda s: spacing_rate * np.exp(s) + decay_rate

    for i, t_i in enumerate(times[1:] + [0]):
        strength = 0
        for j, t_j in enumerate(times[:i+1]):
            strength += (t_j - t_i) ** -get_decay(strengths[j])
        strengths.append(np.log(strength))

    return strengths[-1]


def retrieval_prob(strength, tau=-0.704, s=0.255):
    """Returns the probability of item retrieval based on the *strength*
    of the memory.

    :param strength: Strength of the memory.
    :type strength: float
    """
    val = (tau - strength) / s
    return 1 / (1 + np.exp(val))


def automaticity_level(t):
    """Calculates the level of automaticity (the effort the user had to
    make to retrieve an item from memory) based on response time. The values
    of parameters are based on some statistical experiments.

    :param t: Response time in seconds.
    :type t: float or int
    """
    result = ((4.07446031e-03 * t) -
              (1.18475468e+00 * t ** (1 / 2)) -
              (1.01545130e-05 * t ** (3 / 2)) +
              (4.68002306e+00 * t ** (1 / 3)))
    return max(result - 12.23, 0)


def merge_dicts(*dicts):
    """Merges multiple *dicts* into one.

    :param *dicts: Dictionaries given as positional arguments.
    """
    items = []
    for d in dicts:
        items.extend(d.items())
    merged = {}
    for key, value in items:
        merged[key] = value
    return merged


def time_diff(datetime1, datetime2):
    """Returns difference between the arguments `datetime1` and
    `datetime2` in seconds.

    :type datetime1: string or datetime
    :type datetime2: string or datetime
    """
    if isinstance(datetime1, basestring):
        datetime1 = to_datetime(datetime1)
    if isinstance(datetime2, basestring):
        datetime2 = to_datetime(datetime2)
    return (datetime1 - datetime2).total_seconds()


def to_datetime(date_str):
    """Deserializes given datetime.

    :param date_str: DateTime given as string.
    :type date_str: str
    """
    return datetime.strptime(date_str, config.DATETIME_FORMAT)


def reverse_enumerate(values):
    """Enumerate over an iterable in reverse order while
    retaining proper indexes.

    :param values: List of objects.
    :type values: iterable
    """
    count = len(values)
    for value in values:
        count -= 1
        yield count, value


def timezones(prefix=''):
    """Lists all timezones and alpha2 code of the country.

    :param prefix: Filter timezones with prefix (default is no prefix).
    :type prefix: string
    :return: Dictonary of timezones and country codes.
    :rtype: dict
    """
    timezone_country = {}
    for countrycode in pytz.country_timezones:
        timezones = pytz.country_timezones[countrycode]
        for timezone in timezones:
            if timezone.startswith(prefix):
                timezone_country[timezone] = countrycode
    return timezone_country


def get_places(place_type, places=None, place_types=None):
    """Returns ids of requested type.

    :param place_type: ID of the place type or name.
    :type place_type: int or string
    """
    try:
        type_id = int(place_type)
    except ValueError:
        if place_types is None:
            place_types = load_place_types()
        type_id = place_types.ix[place_type].values[0]

    if places is None:
        places = load_places()
    return set(places[places['type'] == type_id].index)


def get_places_by_prefix(prefix='', places=None):
    """Returns ids of all countries by the given timezone prefix.

    :param prefix: Timezone prefix. Default is no prefix (ids of all
        countries will be returned).
    :type prefix: string
    :rtype: set
    """
    if places is None:
        places = load_places()
    codes = {
        place['code'].upper(): place_id
        for place_id, place in places.T.to_dict().items()
    }
    result = set()
    for timezone, code in timezones(prefix).items():
        if code in codes:
            result.add(codes[code])
    return result


def to_place_name(place_id, places=None):
    """Takes ID of a place and returns its English name.

    :param place_id: ID of the place.
    :type place_id: integer
    :param places: Dictionary of places can be given so that
        there is no need to load it every time the function is called
        (e.g. in a for-loop).
    :type places: dict
    :rtype: string
    """
    places = places or load_places().T.to_dict()
    return unicode(places[place_id]['name']).decode('utf-8')


def connect_points(points):
    """Connects points by linear functions.

    :param vector: Vector containing points to be connected by
        linear functions.
    :param vals: Values valid for each point.
    """
    intervals = {}
    formula = '({{{4}}} - {0}) * ({3} - {2}) / ({1} - {0}) + {2}'
    for i in range(len(points)-1):
        (x1, y1), (x2, y2) = points[i], points[i+1]
        intervals[x1, x2] = formula.format(x1, x2, y1, y2, 'x')
    interval_dict = intervaldict(intervals)

    def linear_fit(x):
        return eval(interval_dict[x].format(x=x))
    return linear_fit


def to_place_type(place_id, places=None, place_types=None):
    """Takes ID of a place and returns its type name.

    :param place_id: ID of the place.
    :type place_id: integer
    :param places: Dictionary of places can be given so that
        there is no need to load it every time the function is called
        (e.g. in a for-loop).
    :type places: dict
    :param place_types: Dictionary of place types can be given so that
        there is no need to load it every time the function is called
        (e.g. in a for-loop).
    :type place_types: dict
    :rtype: string
    """
    places = places or load_places().T.to_dict()
    place_type_id = places[place_id]['type']

    place_types = place_types or load_place_types(index_col='id').T.to_dict()
    return unicode(place_types[place_type_id]['name']).decode('utf-8')


class cached_property(object):
    """A decorator that converts a method into a property. The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value::

        class Person(object):

            @cached_property
            def stats(self):
                # calculate something important here
                return Stats(self.domain)

    The class has to have a `__dict__` in order for this property to
    work.
    """

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__)
        if value is None:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


class keydefaultdict(defaultdict):
    """Defaultdict that takes inserted key as an argument.

    Example::

        d = keydefaultdict(C)
        d[x]  # returns C(x)

    """

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            args = key if isinstance(key, tuple) else (key,)
            ret = self[key] = self.default_factory(*args)
            return ret


class intervaldict(dict):
    """Dictionary that expects intervals of length two as keys.
    Items from the dictionary are retrieved according to the interval
    value.

    Example::

        d = intervaldict({(1, 2): 'a', (8, 10): 'b'})
        d[8.2]  # returns 'b'
        d[1]  # returns 'a'

    """

    def get_interval(self, key):
        if isinstance(key, tuple):
            return key
        for k, v in self.items():
            if k[0] <= key < k[1]:
                return k
        raise KeyError('Key {!r} is not between any values in '
                       'the intervaldict'.format(key))

    def __getitem__(self, key):
        interval = self.get_interval(key)
        return dict.__getitem__(self, interval)

    def __setitem__(self, key, value):
        interval = self.get_interval(key)
        dict.__setitem__(self, interval, value)

    def __contains__(self, key):
        try:
            return bool(self[key]) or True
        except KeyError:
            return False
