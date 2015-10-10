# -*- coding: utf-8 -*-

"""
Selected Filters
================

Some selected filters of data (e.g. European countries, Czech rivers).

"""

from datetime import timedelta

from . import tools


_cache = {}


def place_type(data, type_id):
    """Filters data by any given place type.

    :param data: Data to filter.
    :param type_id: Type of place either by ID or name.
    """
    if 'places' not in _cache:
        _cache['places'] = tools.load_places()
    if 'place_types' not in _cache:
        _cache['place_types'] = tools.load_place_types()

    places = tools.get_places(type_id, **_cache)
    return data['place_id'].isin(places)


def timezone_prefix(data, prefix):
    """Filters data by any country's timezone prefix.

    :param data: Data to filter.
    :param prefix: Country's timezone prefix.
    """
    if 'places' not in _cache:
        _cache['places'] = tools.load_places()

    places = tools.get_places_by_prefix(prefix, places=_cache['places'])
    return data['place_id'].isin(places)


def countries(data):
    """List of world countries."""
    return place_type(data, 'country')


def cities(data):
    """List of cities."""
    return place_type(data, 'city')


def open_questions(data):
    """List only open questions."""
    return data['options'].apply(len) == 0


def european_countries(data):
    """List of European countries."""
    return timezone_prefix(data, 'Europe')


def african_countries(data):
    """List of African countries."""
    return timezone_prefix(data, 'Africa')


def asian_countries(data):
    """List of Asian countries."""
    return timezone_prefix(data, 'Asia')


def american_countries(data):
    """List of American countries."""
    return timezone_prefix(data, 'America')


def usa_states(data):
    """List of USA states."""
    if 'places' not in _cache:
        _cache['places'] = tools.load_places()

    places = {
        idx for idx, place in _cache['places'].T.to_dict().items()
        if place['code'].startswith('us-')
    }
    return data['place_id'].isin(places)


def user_answered(data, condition):
    """Only users with the total number answers according to the
    :fun:`condition`.
    """
    assert callable(condition), "The condition is not a function."
    user_values = data['user_id'].value_counts()
    users_with_answers = user_values[condition(user_values)]
    return data['user_id'].isin(users_with_answers.index)


def place_answered(data, condition):
    """Only places with the total number of answers according to the
    :fun:`condition`.
    """
    assert callable(condition), "The condition is not a function."
    place_values = data['place_id'].value_counts()
    places_with_answers = place_values[condition(place_values)]
    return data['place_id'].isin(places_with_answers.index)


def user_item_answered(data, condition):
    """Only the items with the total number of answers according to the
    :fun:`condition`.
    """
    assert callable(condition), "The condition is not a function."

    items = data['user_id'].map(str) + ',' + data['place_id'].map(str)
    item_values = items.value_counts()

    items_with_answers = item_values[condition(item_values)]
    return items.isin(items_with_answers.index)


def for_staircase(data):
    """Data suitable for calculating staircase function."""
    return (
        user_answered(data, lambda x: x >= 20)
        &
        place_answered(data, lambda x: x >= 50)
        &
        user_item_answered(data, lambda x: x >= 3)
    )


def sequentize(data, delta=timedelta(days=5)):
    """Creates sequences of answers where the timedelta between
    the first and the last answer to an item is at least as big as
    specified by the parameter ``delta``.
    """
    groups = data.sort(['inserted']).groupby(['user_id', 'place_id'])
    first, last = groups.first(), groups.last()

    filtered = (last['inserted'] - first['inserted'])
    filtered = filtered[filtered > delta]

    users = filtered.index.get_level_values('user_id')
    places = filtered.index.get_level_values('place_id')

    return data['user_id'].isin(users) & data['place_id'].isin(places)


def classmates(data, minimum=10):
    """Filters data so that only classmates are counted
    (i.e. the answers of students who use the system at school).
    """
    users = set()
    for index, group in data.groupby(['ip_id']):
        classmates = set(group['user_id'].values)
        if len(classmates) >= minimum:
            users |= classmates
    return data['user_id'].isin(users)
