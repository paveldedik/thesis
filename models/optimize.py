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
from .models import EloModel, PFAModel, PFASpacing, PFAStaircase


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


class GradientResult(object):
    """Representation of the result of gradient descent."""

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


class GradientDescent(object):
    """Encapsulates gradient descent for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search(self, model_fun, parameters,
               step_size=1, precision=0.01, maxiter=50):
        """Finds optimal parameters for given model.

        :param model_fun: Callable that is called with ``parameters``.
            Must return gradient.
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

        descent = defaultdict(lambda: [])

        while diff(old_params, new_params) > precision:

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

        return GradientResult(fitted_params, gradients)

    def search_pfa(self, init_gamma, init_delta, **search_kwargs):
        """Finds optimal parameters for the PFAModel.

        :param init_gamma: Initial gamma value.
        :param init_delta: Initial delta value.
        :param **search_kwargs: Optional parameters passed to the
            method :meth:`GradientDescent.serach`.
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
            method :meth:`GradientDescent.serach`.
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
            method :meth:`GradientDescent.serach`.
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
