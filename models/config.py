# -*- coding: utf-8 -*-

"""
Configurations
==============

"""

import os


### Developer-specific configurations

DATA_HOME = '/home/pavel/Projects/thesis/data/'

#: Path to the CSV file containing all answers.
DATA_ANSWERS_PATH = os.path.join(DATA_HOME, 'answers.csv')

#: Path to the CSV file containing all places.
DATA_PLACES_PATH = os.path.join(DATA_HOME, 'places.csv')

#: Path to the CSV file containing all place types.
DATA_PLACE_TYPES_PATH = os.path.join(DATA_HOME, 'place_type.csv')

#: Path to the CSV file containing users (see :func:`generate_users`).
DATA_USERS_PATH = os.path.join(DATA_HOME, 'users.csv')


### Other configurations

#: Columns currently not used in models. They are ignored in
#: the :func:`prepare_data` so that the memory requirements are lower.
IGNORED_COLUMNS = [
    'place_map', 'language', 'ip_country',
]

#: Columns renamed in models.
ANSWERS_COLUMNS = [
    'id', 'user_id', 'place_id', 'place_answered', 'type', 'inserted',
    'response_time', 'place_map', 'language', 'options', 'ip_country', 'ip_id',
]

#: Names of columns in generated CSV containing users.
USERS_COLUMNS = [
    'user_id', 'first_answer_id', 'first_answer_inserted'
]

#: DateTime format of the field `inserted`.
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
