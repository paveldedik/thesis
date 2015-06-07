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
