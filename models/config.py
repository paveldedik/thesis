# -*- coding: utf-8 -*-

"""
Configurations
==============

"""

### Developer-specific configurations

#: Path to the CSV file containing all answers.
DATA_ANSWERS_PATH = '/home/pavel/Projects/thesis/data/answers.csv'

#: Path to the CSV file containing all places.
DATA_PLACES_PATH = '/home/pavel/Projects/thesis/data/places.csv'

#: Path to the CSV file containing users (see :func:`generate_users`).
DATA_USERS_PATH = '/home/pavel/Projects/thesis/data/users.csv'


### Other configurations

#: Columns currently not used in models. They are ignored in
#: the :func:`prepare_data` so that the memory requirements are lower.
IGNORED_COLUMNS = [
    'place_map', 'language', 'options'
]

#: Columns renamed in models.
RENAMED_COLUMNS = {
    'user': 'user_id',
    'place_asked': 'place_id',
}

#: Names of columns in generated CSV containing users.
USERS_COLUMNS = [
    'user_id', 'first_answer_id', 'first_answer_inserted'
]

#: DateTime format of the field `inserted`.
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
