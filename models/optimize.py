# -*- coding: utf-8 -*-

"""
Optimization Methods
====================

"""

from __future__ import division

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import optimize

from . import tools
from .tests import PerformanceTest
from .models import EloModel, PFAModel, PFASpacing


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
            [result.rmse().value for result in row]
            for row in self.grid
        ])

    @tools.cached_property
    def auc(self):
        """Grid Search errors estimations using AUC."""
        return np.array([
            [result.auc().value for result in row]
            for row in self.grid
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

            grid[y, x] = test
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
            return test.rmse().value

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
            return test.rmse().value

        return optimize.minimize(fun, [gamma, delta])


class GradientDescent(object):
    """Encapsulates gradient descent for various models.

    :param data: Data with answers in a DataFrame.
    """

    def __init__(self, data):
        self.data = data

    def search_pfa(self, init_gamma, init_delta,
                   step_size=0.01, precision=0.01, maxiter=50):
        """Finds optimal parameters for the PFAModel.

        :param init_gamma: Initial gamma value.
        :param init_delta: Initial delta value.
        :param step_size: Step size. Default is :num:`0.01`.
        :param precision: The algorithm stops iterating when the precision
            gets below this value. Default is :num:`0.01`.
        :param maxiter: Maximum number of iteration. Default is :num:`50`.
        """
        def pfa_off(x, y):
            elo = EloModel()
            pfa = PFAModel(elo, gamma=x, delta=y)
            pfa_test = PerformanceTest(pfa, self.data)

            pfa_test.run()
            return pfa_test.pred_off()

        gamma_val = delta_val = pfa_off(init_gamma, init_delta)

        old_gamma = init_gamma
        old_delta = init_delta

        for _ in range(maxiter):

            # import pdb; pdb.set_trace()
            new_gamma = old_gamma + step_size * gamma_val
            new_delta = old_delta - step_size * delta_val

            gamma_val = pfa_off(new_gamma, old_delta)
            delta_val = pfa_off(old_gamma, new_delta)

            old_gamma = new_gamma
            old_delta = new_delta

            tools.echo('gamma:{}/delta:{}'.format(new_gamma, new_delta))
