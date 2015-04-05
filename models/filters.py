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
    places = tools.get_codes()
    return data['place_id'].isin(places)


def european_countries(data):
    """List of European countries."""
    places = tools.get_codes('Europe')
    return data['place_id'].isin(places)


def africa_countries(data):
    """List of African countries."""
    places = tools.get_codes('Africa')
    return data['place_id'].isin(places)


def asia_countries(data):
    """List of Asian countries."""
    places = tools.get_codes('Asia')
    return data['place_id'].isin(places)


def america_countries(data):
    """List of American countries."""
    places = tools.get_codes('America')
    return data['place_id'].isin(places)


def usa_states(data):
    """List of USA states."""
    places = {
        idx for idx, place in tools.load_places().T.to_dict().items()
        if place['code'].startswith('us-')
    }
    return data['place_id'].isin(places)
