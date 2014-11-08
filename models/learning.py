# -*- coding: utf-8 -*-


import os
import time
import json
from datetime import datetime, timedelta

import click
import model
from scipy import optimize


DB = {
    'Q.E.D.': 'quod erat demonstrandum',
    'R.I.P.': 'requiescat in pace',
    'i.e.': 'id est',
    'e.g.': 'exempli gratia',
    'p.m.': 'post meridiem',
    'a.m.': 'ante meridiem',
    'A.D.': 'anno domini',
    'et al.': 'et alii',
    'etc.': 'et cetera',
}


def load_db(path='../data/learning.json'):
    with open(path, 'r') as fd:
        data = fd.read()
    return json.loads(data)


def dump_db(data, path='../data/learning.json'):
    json_data = json.dumps(data, indent=4)
    with open(path, 'w') as fd:
        fd.write(json_data)


class Tutor(object):

    reviews_file = './learning.json'
    datetime_format = '%Y-%m-%d %H:%M:%S.%f'
    question_message = 'What does the abbreviation "{}" stand for?'

    def __init__(self, db):
        self.db = db
        self.reviews = {key: [] for key in db}

    @property
    def strengths(self):
        strengths = {}
        for abbr in self.db:
            timedeltas = self.reviews_delta[abbr]
            strengths[abbr] = model.memory_strength(timedeltas)
        return strengths

    @property
    def reviews_delta(self):
        result = {}
        for abbr, reviews in self.reviews.items():
            result[abbr] = [self.to_seconds(rev) for rev in reviews]
        return result

    def next_run(self, abbr):
        reviews = self.reviews_delta[abbr]

        def fun(x):
            return model.memory_strength([rev + x for rev in reviews])

        try:
            seconds = optimize.newton(fun, 20-min(reviews))
            return datetime.now() + timedelta(seconds=seconds), seconds
        except RuntimeError:
            return None, None

    def to_seconds(self, date_str):
        now = datetime.now()
        review = datetime.strptime(date_str, self.datetime_format)
        return (now - review).total_seconds()

    def load_reviews(self):
        self.reviews = load_db(path=self.reviews_file)
        self.write('DB loaded.')

    def save_reviews(self):
        dump_db(self.reviews, self.reviews_file)
        self.write('DB saved.')

    def write(self, msg, delay=.1):
        print msg
        time.sleep(delay)

    def read(self, msg, delay=.5):
        answer = raw_input(msg)
        time.sleep(delay)
        return answer

    def learn(self, question):
        now = datetime.strftime(datetime.now(), self.datetime_format)
        self.reviews[question].append(now)

    def select_question(self):
        return min(self.strengths, key=lambda k: self.strengths[k])

    def format_knowledge(self):
        msg = 'Current knowledge:'
        for abbr in sorted(self.strengths,
                           key=lambda x: self.strengths[x],
                           reverse=True):
            strength = self.strengths[abbr]
            reviews = len(self.reviews[abbr])
            next_run, _ = self.next_run(abbr)
            string = '\n {0: <7}: {1}, {2} reviews, next run: {3}'
            msg += string.format(abbr, strength, reviews, next_run)
        return msg

    def run(self):
        print 'Program is running...'
        while True:
            question = self.select_question()

            if self.strengths[question] > 0:
                next_run, seconds = self.next_run(question)
                self.write('All items are learned, '
                           'next run at {}'.format(next_run))
                time.sleep(seconds)
                continue

            self.write('\n' + self.question_message.format(question))

            answer = self.read('Your answer: ')
            correct_answer = self.db[question]

            if answer.lower() == correct_answer.lower():
                self.write('The answer is correct.')
                self.learn(question)
            else:
                self.write('The answer is incorrect.')
                self.write('The correct answer is "{}"'
                           '.'.format(correct_answer), delay=2)


@click.command()
@click.option('--knowledge', is_flag=True)
def main(knowledge):
    tutor = Tutor(DB)

    if os.path.isfile(tutor.reviews_file):
        tutor.load_reviews()
    if knowledge:
        tutor.write('\n' + tutor.format_knowledge() + '\n', delay=0)
        return
    else:
        try:
            tutor.run()
        except KeyboardInterrupt:
            tutor.save_reviews()
            print 'Terminating...'


if __name__ == '__main__':
    main()
