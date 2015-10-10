# -*- coding: utf-8 -*-

"""
Optimization Methods
====================

"""

from __future__ import division

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize

from . import tools
from .tests import PerformanceTest
from .models import EloModel, PFAModel, PFASpacing, PFAStaircase, PFAGong


class GridResult(object):
    """Represents a GRID search result.

    :param grid: A matrix representing the results of the search.
    :type grid: :class:`numpy.matrix`
    :param xlabel: Name of the x-axis.
    :type xlabel: str
    :param ylavel: Name of the y-axis.
    :type ylabel: str
    :param xvalues: Values on the x-axis.
    :type xvalues: list
    :param yvalues: Values on the y-axis.
    :type yvalues: list
    """

    def __init__(self, grid,
                 xlabel=None, ylabel=None,
                 xvalues=None, yvalues=None):
        self.grid = grid
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xvalues = xvalues
        self.yvalues = yvalues

        self.extent = np.array([
            min(self.xvalues), max(self.xvalues),
            max(self.yvalues), min(self.yvalues),
        ])

    @tools.cached_property
    def rmse(self):
        """Grid Search errors estimations using RMSE."""
        return np.array([
            [result.rmse for result in row] for row in self.grid
        ])

    @tools.cached_property
    def auc(self):
        """Grid Search errors estimations using AUC."""
        return np.array([
            [result.auc for result in row] for row in self.grid
        ])

    @tools.cached_property
    def off(self):
        """Grid Search errors estimations using the average of
        ``predicted - observerd``.
        """
        return np.array([
            [result.off for result in row] for row in self.grid
        ])

    @tools.cached_property
    def rmse_best(self):
        """Values of `xvalues` and `yvalues` with best RMSE."""
        minimum = np.unravel_index(self.rmse.argmin(), self.rmse.shape)
        return np.array([self.xvalues[minimum[1]], self.yvalues[minimum[0]]])

    @tools.cached_property
    def auc_best(self):
        """Values of `xvalues` and `yvalues` with best AUC."""
        maximum = np.unravel_index(self.auc.argmax(), self.auc.shape)
        return np.array([self.xvalues[maximum[1]], self.yvalues[maximum[0]]])

    def _plot_grid(self, grid, **img_kwargs):
        """Plots the result of the GRID search.
        Uses :func:`~matplotlib.pyplot.imshow` to plot the data.

        :param grid: The grid to plot.
        :type grid: list of lists or :class:`numpy.matrix`.
        :param **img_kwargs: Key-word arguments passed to the
            :func:`~matplotlib.pyplot.imshow`.
        """
        img_kwargs.setdefault('cmap', cm.Greys)
        img_kwargs.setdefault('interpolation', 'nearest')
        img_kwargs.setdefault('extent', self.extent)
        img_kwargs.setdefault('aspect', 'auto')

        img_title = img_kwargs.pop('title', 'Grid Search')
        img_xlabel = img_kwargs.pop('xlabel', self.xlabel)
        img_ylabel = img_kwargs.pop('ylabel', self.ylabel)

        plot = plt.imshow(grid, **img_kwargs)
        plt.colorbar(plot)
        plt.xlabel(img_xlabel)
        plt.ylabel(img_ylabel)
        plt.title(img_title)
        plt.show()

        return plot

    def plot_rmse(self, **img_kwargs):
        """Plots the result of the GRID search.
        Uses :func:`~matplotlib.pyplot.imshow` to plot the data.

        :param **img_kwargs: Key-word arguments passed to the
            :func:`~matplotlib.pyplot.imshow`.
        """
        img_kwargs.setdefault('title', 'Grid Search, metric: RMSE')
        img_kwargs.setdefault('cmap', cm.Greys_r)
        return self._plot_grid(self.rmse, **img_kwargs)

    def plot_auc(self, **img_kwargs):
        """Plots the result of the GRID search.
        Uses :func:`~matplotlib.pyplot.imshow` to plot the data.

        :param **img_kwargs: Key-word arguments passed to the
            :func:`~matplotlib.pyplot.imshow`.
        """
        img_kwargs.setdefault('title', 'Grid Search, metric: AUC')
        return self._plot_grid(self.auc, **img_kwargs)

    def plot_off(self, **img_kwargs):
        """Plots the result of the GRID search.
        Uses :func:`~matplotlib.pyplot.imshow` to plot the data.

        :param **img_kwargs: Key-word arguments passed to the
            :func:`~matplotlib.pyplot.imshow`.
        """
        img_kwargs.setdefault('title',
                              'Grid Search, metric: observed - predicted')
        return self._plot_grid(self.off, **img_kwargs)

    def plot(self):
        """Plots the result of the GRID search.
        Uses :func:`~matplotlib.pyplot.imshow` to plot the data.
        """
        plt.figure(1)

        plt.subplot(121)
        plot1 = self.plot_rmse()

        plt.subplot(122)
        plot2 = self.plot_auc()

        return [plot1, plot2]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            'RMSE:\n min: {0}\n {1}'
            '\n\n'
            'AUC:\n min: {2}\n {3}'
        ).format(
            self.rmse_min.round(3),
            self.rmse.round(3),
            self.auc_min.round(3),
            self.auc.round(3),
        )


class DescentResult(object):
    """Representation of the result of NaiveDescent."""

    def __init__(self, params, grads):
        self.params = pd.DataFrame(params)
        self.grads = pd.Series(grads)
        self.iterations = len(self.grads)

    @property
    def best(self):
        """The best fitted parameters."""
        return self.params.iloc[-1]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            'Iterations: {}\n'
            'Best:\n{}'
        ).format(
            self.iterations,
            self.best.round(3),
        )


class GradientResult(object):
    """Representation of the result of GradientDescent."""

    def __init__(self, model, parameters):
        self.model = model
        self.parameters = parameters
        self.iterations = range(len(parameters))

        self.deltas = [params['delta'] for params in self.parameters]
        self.gammas = [params['gamma'] for params in self.parameters]
        self.staircases = [params['staircase'] for params in self.parameters]

        self.intervals = list(sorted(i for i in self.staircases[-1]))

    @property
    def best(self):
        """The best fitted parameters."""
        return {
            'gamma': self.gammas[-1],
            'delta': self.deltas[-1],
            'staircase': self.staircases[-1],
        }

    def plot(self, **kwargs):
        """Plots the result of the gradient descent.
        Uses :func:`~matplotlib.pyplot.plot` to plot the data.

        :param **kwargs: Key-word arguments passed to the
            :func:`~matplotlib.pyplot.plot`.
        """
        results = sorted(self.staircases[-1].items(), key=lambda x: x[0])
        staircase_times = self.model.metadata['staircase_times']

        x_axis = [np.mean(staircase_times[i]) for i in self.intervals]
        y_axis = [value for interval, value in results]

        xlabel = kwargs.pop('xlabel', 'Time from previous attempt in seconds.')
        ylabel = kwargs.pop('ylabel', 'Memory activation')
        title = kwargs.pop('title', '')

        plot = plt.plot(x_axis, y_axis, '.-', **kwargs)
        plt.xscale('log')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        return plot

    def format_staircases(self, indexes=None):
        """Formats staircase function in a readable way.

        :param indexes: Staircases to show (referenced by the index).
            `[-1]` formats only the last staircase values. By default,
            all staircase values are formated.
        """
        indexes = indexes or self.iterations
        staircases = [self.staircases[i] for i in indexes]

        ranges = sorted([x[1] for x in staircases[0]])
        head = ('{:9.0f}' * len(staircases[0])).format(*ranges)

        body = ''
        for staircase in staircases:
            stair = list(sorted(staircase.items(), key=lambda x: x[0]))
            body += ('{:+9.3f}' * len(stair)).format(*[v for k, v in stair])
            body += '\n'

        return '{}\n{}'.format(head, body)

    def __repr__(self):
        return (
            'Iterations: {0}\n'
            'Gamma: {1:.5f}\n'
            'Delta: {2:.5f}\n'
            'Staircase:\n{3}'
        ).format(
            len(self.iterations)-1,
            self.best['gamma'],
            self.best['delta'],
            self.format_staircases([-1])
        )


class GridSearch(object):
    """Encapsulates GRID searches for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search(self, factory, xvalues, yvalues, **result_kwargs):
        """Performes grid search on ELO model using given parameters.

        :param factory: Model facotry which is used to instantiate
            model with all the combinations of `xvalues` and `yvalues`.
        :type factory: callable
        :param xvalues: List of values for first positional argument
            passed on to the model factory.
        :type xvalues: iterable
        :param yvalues: List of values for second positional argument
            passed on to the model factory.
        :type yvalues: iterable
        :param **result_kwargs: Optional arguments passed on to
            the :class:`GridResult` instance.
        """
        m, n = len(xvalues), len(yvalues)
        grid = np.array([[None] * m] * n)

        for x, y in itertools.product(range(m), range(n)):
            model = factory(xvalues[x], yvalues[y])
            test = PerformanceTest(model, self.data)
            test.run()

            grid[y, x] = test.results['train']
            tools.echo('{}/{} {}/{}'.format(x+1, m, y+1, n))

        return GridResult(
            grid=grid,
            xvalues=xvalues,
            yvalues=yvalues,
            **result_kwargs
        )

    def search_elo(self, alphas, betas):
        """Performes grid search on ELO model using given parameters.

        :param alphas: Alpha parameters (see :class:`EloModel`).
        :type alphas: list or :class:`numpy.array`
        :param betas: Beta paramters (see :class:`EloModel`).
        :type betas: list or :class:`numpy.array`
        """
        def elo_factory(x, y):
            return EloModel(alpha=x, beta=y)

        return self.search(
            factory=elo_factory,
            xvalues=alphas,
            yvalues=betas,
            xlabel='alpha',
            ylabel='beta',
        )

    def search_pfa(self, gammas, deltas):
        """Performes grid search on PFA extended model using given parameters.

        :param gammas: Gamma parameters (see :class:`PFAModel`).
        :type gammas: list or :class:`numpy.array`
        :param deltas: Delta paramters (see :class:`PFAModel`).
        :type deltas: list or :class:`numpy.array`
        """
        def pfa_factory(x, y):
            elo = EloModel()
            return PFAModel(elo, gamma=x, delta=y)

        return self.search(
            factory=pfa_factory,
            xvalues=gammas,
            yvalues=deltas,
            xlabel='gammas',
            ylabel='deltas',
        )

    def search_pfas(self, decays, spacings):
        """Performes grid search on PFA extended with spacing and forgetting
        using given parameters.

        :param decays: Decay rates (see :class:`PFASpacing`).
        :type decays: list or :class:`numpy.array`
        :param spacings: Spacing rates (see :class:`PFASpacing`).
        :type spacings: list or :class:`numpy.array`
        """
        def pfas_factory(x, y):
            elo = EloModel()
            return PFASpacing(elo, decay_rate=x, spacing_rate=y)

        return self.search(
            factory=pfas_factory,
            xvalues=decays,
            yvalues=spacings,
            xlabel='decay rates',
            ylabel='spacing rates',
        )


class RandomSearch(object):
    """Encapsulates random searches for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search_elo(self, alpha, beta):
        """Performes random search on ELO model using given initial
        parameters.

        :param alpha: Initial alpha value (see :class:`EloModel`).
        :type alpha: float
        :param beta: Initial beta value (see :class:`EloModel`).
        :type beta: float
        """
        def fun(x):
            elo = EloModel(alpha=x[0], beta=x[1])
            test = PerformanceTest(elo, self.data)

            test.run()

            tools.echo('alpha={x[0]} beta={x[1]}'.format(x=x))
            return test.results['train'].rmse

        return optimize.minimize(fun, [alpha, beta])

    def search_pfa(self, gamma, delta):
        """Performes random search on ELO model using given initial
        parameters.

        :param gamma: Initial gamma value (see :class:`PFAModel`).
        :type gamma: float
        :param delta: Initial delta value (see :class:`PFAModel`).
        :type delta: float
        """
        elo = EloModel()

        def fun(x):
            pfa = PFAModel(elo, gamma=x[0], delta=x[1])
            test = PerformanceTest(pfa, self.data)

            test.run()

            tools.echo('gamma={x[0]} delta={x[1]}'.format(x=x))
            return test.results['train'].rmse

        return optimize.minimize(fun, [gamma, delta])


class NaiveDescent(object):
    """Encapsulates the modified gradient descent (which is not in fact
    based on the partial derivatives of a function) for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search(self, model_fun, parameters,
               step_size=1, precision=0.01, maxiter=50):
        """Finds optimal parameters for given model.

        :param model_fun: Callable that trains the model on the given
            parameters.
        :param parameters: Dictionary of parameters to fit.
        :param step_size: Step size. Default is :num:`0.01`.
        :param precision: The algorithm stops iterating when the precision
            gets below this value. Default is :num:`0.01`.
        :param maxiter: Maximum number of iteration. Default is :num:`50`.
        """
        def diff(old, new):
            return sum(abs(old[key] - new[key]) for key in new)

        old_params = {p: np.inf for p in parameters}
        new_params = dict(parameters)

        grad = model_fun(**new_params)
        grads = {p: grad for p in parameters}

        iterations = 0
        descent = defaultdict(lambda: [])

        while (diff(old_params, new_params) > precision
               and iterations < maxiter):

            iterations += 1
            old_params = dict(new_params)

            for key in parameters:
                value = old_params[key] - step_size * grads[key]
                params = tools.merge_dicts(old_params, {key: value})

                grads[key] = model_fun(**params)
                new_params[key] = value

                descent[key].append(new_params[key])

            grads_mean = np.average(grads.values())
            descent['grad'].append(grads_mean)

            msg = '\n'.join([
                '{}: {}; grad: {}'.format(key, val, grads[key])
                for key, val in new_params.items()
            ])
            tools.echo(msg)

        gradients = descent.pop('grad')
        fitted_params = descent

        return DescentResult(fitted_params, gradients)

    def search_pfa(self, init_gamma, init_delta, **search_kwargs):
        """Finds optimal parameters for the PFAModel.

        :param init_gamma: Initial gamma value.
        :param init_delta: Initial delta value.
        :param **search_kwargs: Optional parameters passed to the
            method :meth:`NaiveDescent.serach`.
        """
        def pfa_fun(gamma, delta):
            elo = EloModel()
            pfa = PFAModel(elo, gamma=gamma, delta=delta)
            pfa_test = PerformanceTest(pfa, self.data)

            pfa_test.run()
            return pfa_test.results['train'].off

        parameters = {
            'gamma': init_gamma, 'delta': init_delta
        }
        return self.search(pfa_fun, parameters, **search_kwargs)

    def search_pfag(self, init_gamma, init_delta, **search_kwargs):
        """Finds optimal parameters for the PFAGong model.

        :param init_gamma: Initial gamma value.
        :param init_delta: Initial delta value.
        :param **search_kwargs: Optional parameters passed to the
            method :meth:`NaiveDescent.serach`.
        """
        def pfag_fun(gamma, delta):
            elo = EloModel()
            pfag = PFAGong(elo, gamma=gamma, delta=delta)
            pfag_test = PerformanceTest(pfag, self.data)

            pfag_test.run()
            return pfag_test.results['train'].off

        parameters = {
            'gamma': init_gamma, 'delta': init_delta
        }
        return self.search(pfag_fun, parameters, **search_kwargs)

    def search_staircase(self, init_gamma, init_delta, init_staircase,
                         **search_kwargs):
        """Finds optimal parameters for the `PFAStaircase` model.

        :param init_gamma: Initial gamma value.
        :type init_gamma: int or float
        :param init_delta: Initial delta value.
        :type init_delta: int or float
        :param init_staircase: Initial staircase function.
        :type init_staircase: dict
        :param **search_kwargs: Optional parameters passed to the
            method :meth:`NaiveDescent.serach`.
        """
        interval, init_value = init_staircase.items()[0]

        def pfast_fun(gamma, delta, staircase_value):
            elo = EloModel()
            staircase = {interval: staircase_value}

            pfa = PFAStaircase(elo, gamma=gamma, delta=delta,
                               staircase=staircase)
            pfa_test = PerformanceTest(pfa, self.data)

            pfa_test.run()
            return pfa_test.results['train'].off

        parameters = {
            'gamma': init_gamma,
            'delta': init_delta,
            'staircase_value': init_value,
        }
        return self.search(pfast_fun, parameters, **search_kwargs)

    def search_staircase_only(self, init_staircase, **search_kwargs):
        """Finds optimal parameters for the `PFAStaircase` model.

        :param init_staircase: Initial staircase function.
        :type init_staircase: dict
        :param **search_kwargs: Optional parameters passed to the
            method :meth:`NaiveDescent.serach`.
        """
        interval, init_value = init_staircase.items()[0]

        def pfast_fun(staircase_value):
            elo = EloModel()
            staircase = {interval: staircase_value}

            pfa = PFAStaircase(elo, staircase=staircase)
            pfa_test = PerformanceTest(pfa, self.data)

            pfa_test.run()
            return pfa_test.results['train'].off

        parameters = {
            'staircase_value': init_value,
        }
        return self.search(pfast_fun, parameters, **search_kwargs)


class GreedySearch(object):
    """Similar to the gradient descent method but searches for
    the optimum of a selected objective function.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search(self, model_fun, init_parameters, init_epsilons,
               altitude_ratio=1, precision=0.001, maxiter=50):
        """Finds optimal parameters for given model function.

        :param model_fun: Callable that trains the model on the given
            parameters.
        :param init_parameters: Dictionary of parameters to fit.
        :param init_epsilons: Dictionary of initial values for the
            evaluation of the parameter's neigbourhood.
        :param altitude_ratio: The ratio of the change in altitude.
            Higher value means that the change in altitude (epsilon)
            is bigger with each iteration. Default is :num:`1`.
        :param precision: The algorithm stops iterating when the precision
            gets below this value. Default is :num:`0.001`.
        :param maxiter: Maximum number of iteration. Default is :num:`50`.
        """
        def diff(old, new):
            return sum(abs(old[key] - new[key]) for key in new)

        epsilons = dict(init_epsilons)
        parameters = dict(init_parameters)

        for iteration in xrange(1, maxiter+1):
            altitude = model_fun(**parameters)
            new_parameters = {}

            for name, value in parameters.items():
                positive = value + epsilons[name]
                negative = value - epsilons[name]

                positive_p = tools.merge_dicts(parameters, {name: positive})
                negative_p = tools.merge_dicts(parameters, {name: negative})

                altitudes = {
                    positive: model_fun(**positive_p),
                    negative: model_fun(**negative_p),
                    value: altitude,
                }
                best = min(altitudes, key=lambda x: altitudes[x])
                new_parameters[name] = best

                change = (altitude - altitudes[best]) * altitude_ratio
                epsilons[name] -= epsilons[name] * change

            old_parameters = parameters
            parameters = new_parameters

            template = 'altitude: {}\nparameters: {}\nepsilons: {}'
            tools.echo(template.format(altitude, parameters, epsilons))

            if diff(old_parameters, parameters) < precision:
                break

        return parameters


class GradientDescent(object):
    """Encapsulates the modified gradient descent (which is not in fact
    based on the partial derivatives of a function) for various models.

    :param data: Data with answers in a DataFrame.
    """

    class PFAStaircaseFit(PFAStaircase):

        def __init__(self, *args, **kwargs):
            self.learn_rate = kwargs.pop('learn_rate', 0.02)

            self.log_metadata = kwargs.pop('log_metadata', False)
            self.log_staircase = kwargs.pop('log_staircase', False)

            self.metadata = {}
            self.random_factor = kwargs.pop('random_factor')
            self.random_chance = kwargs.pop('random_chance', 1000)

            if self.log_metadata:
                self.metadata['diffs'] = []
                if self.log_staircase:
                    self.metadata['staircase_items'] = defaultdict(lambda: 0)
                    self.metadata['staircase_times'] = defaultdict(list)

            super(type(self), self).__init__(*args, **kwargs)

        def update(self, answer):
            """Performes update of current knowledge of a user based on the
            given answer.

            :param answer: Answer to a question.
            :type answer: :class:`pandas.Series` or :class:`Answer`
            """
            item = self.items[answer.user_id, answer.place_id]

            shift = answer.is_correct - self.predict(answer)
            has_practices = bool(item.practices)

            if has_practices:
                seconds = tools.time_diff(answer.inserted, item.last_inserted)
                self.staircase[seconds] += self.learn_rate * shift * 3
            else:
                item.gamma_effect = 0
                item.delta_effect = 0

            self.gamma += self.learn_rate * shift * item.gamma_effect
            self.delta += self.learn_rate * shift * item.delta_effect

            if self.random_factor is not None:
                factor = self.random_factor
                chance = self.random_chance

                if not np.random.randint(chance):
                    self.gamma += np.random.uniform(-factor, factor)
                if not np.random.randint(chance):
                    self.delta += np.random.uniform(-factor, factor)
                if has_practices and not np.random.randint(chance):
                    self.staircase[seconds] += \
                        np.random.uniform(-factor, factor)

            if answer.is_correct:
                item.inc_knowledge(self.gamma * shift)
                item.gamma_effect += shift
            else:
                item.inc_knowledge(self.delta * shift)
                item.delta_effect += shift

            if self.log_metadata:
                self.metadata['diffs'].append(shift)
                if self.log_staircase and has_practices:
                    interval = self.staircase.get_interval(seconds)
                    self.metadata['staircase_items'][interval] += 1
                    self.metadata['staircase_times'][interval] += [seconds]

            item.add_practice(answer)

        def train(self, data):
            """Trains the model on given data set.

            :param data: Data set on which to train the model.
            :type data: :class:`pandas.DataFrame`
            """
            self.prior.train(data)
            super(type(self), self).train(data)

    def __init__(self, data):
        self.data = data

    def search(self, model_fun, init_parameters,
               init_learn_rate=0.01, number_of_iter=10, log_metadata=True,
               echo_iterations=True, random_factor=None, random_chance=None):
        """Finds optimal parameters for given model.

        :param model_fun: Callable that trains the model using the given
            parameters.
        :param parameters: Dictionary of parameters to fit.
        :param init_learn_rate: Initial learning rate Default is :num:`0.01`.
        :param number_of_iter: Number of iteration. Default is :num:`10`.
        :param log_metadata: Whether to log metadata information.
        """
        print_format = '{:10.5f} {:10.5f} {:10.5f}'

        def pretty_echo(p):
            if not echo_iterations:
                return
            string = print_format.format(
                p['gamma'], p['delta'], p.get('off', np.inf))
            tools.echo(string, clear=False)

        pretty_echo(init_parameters)
        parameters = [init_parameters]

        for i in range(1, number_of_iter + 1):
            model_kwargs = {
                'learn_rate': init_learn_rate / (i / 2),
                'log_metadata': log_metadata,
                'log_staircase': i == number_of_iter,
                'random_factor': (random_factor / (i ** 2)
                                  if random_factor else None),
                'random_chance': random_chance or 1000,
            }
            model_kwargs.update(parameters[i-1])

            model = model_fun(**model_kwargs)
            model.train(self.data)

            parameters.append({})
            for param in parameters[i-1]:
                parameters[i][param] = getattr(model, param)

            pretty_echo(dict(gamma=model.gamma, delta=model.delta,
                             off=np.mean(model.metadata['diffs'])))
        return GradientResult(model, parameters)

    def search_staircase(self, init_gamma=2.5, init_delta=0.8,
                         init_staircase=None, **kwargs):
        """Finds optimal parameters for the `PFAStaircase` model.

        :param init_gamma: Initial gamma value.
        :param init_delta: Initial delta value.
        :param init_staircase: Initial staircase function.
        :type init_staircase: dict
        """
        def model_fun(**init_params):
            prior = EloModel()
            return self.PFAStaircaseFit(prior, **init_params)

        parameters = {
            'gamma': init_gamma,
            'delta': init_delta,
            'staircase': init_staircase or dict.fromkeys([
                (0, 60), (60, 90), (90, 150), (150, 300), (300, 600),
                (600, 60*30), (60*30, 60*60*3), (60*60*3, 60*60*24),
                (60*60*24, 60*60*24*5), (60*60*24*5, np.inf),
            ], 0)
        }
        return self.search(model_fun, parameters, **kwargs)
