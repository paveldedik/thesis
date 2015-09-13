# -*- coding: utf-8 -*-

"""
Selected Filters
================

Some selected filters of data (e.g. European countries, Czech rivers).

"""

from . import tools


def open_questions(data):
    """List only open questions."""
    return data['number_of_options'] == 0


def world_countries(data):
    """List of world countries."""
    places = tools.get_places()
    return data['place_id'].isin(places)


def european_countries(data):
    """List of European countries."""
    places = tools.get_places('Europe')
    return data['place_id'].isin(places)


def african_countries(data):
    """List of African countries."""
    places = tools.get_places('Africa')
    return data['place_id'].isin(places)


def asian_countries(data):
    """List of Asian countries."""
    places = tools.get_places('Asia')
    return data['place_id'].isin(places)


def american_countries(data):
    """List of American countries."""
    places = tools.get_places('America')
    return data['place_id'].isin(places)


def usa_states(data):
    """List of USA states."""
    places = {
        idx for idx, place in tools.load_places().T.to_dict().items()
        if place['code'].startswith('us-')
    }
    return data['place_id'].isin(places)


def user_answered(data, condition):
    """Only users with the total number answers according to the
    :fun:`condition`.
    """
    assert callable(condition), "The condition must be a function."
    user_values = data['user_id'].value_counts()
    users_with_answers = user_values[condition(user_values)]
    return data['user_id'].isin(users_with_answers.index)


def place_answered(data, condition):
    """Only places with the total number of answers according to the
    :fun:`condition`.
    """
    assert callable(condition), "The condition must be a function."
    place_values = data['place_id'].value_counts()
    places_with_answers = place_values[condition(place_values)]
    return data['place_id'].isin(places_with_answers.index)


def user_item_answered(data, condition):
    """Only the items with the total number of answers according to the
    :fun:`condition`.
    """
    assert callable(condition), "The condition must be a function."

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
