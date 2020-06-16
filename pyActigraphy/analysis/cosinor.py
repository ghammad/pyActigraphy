import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lmfit import fit_report, minimize, Parameters


def _cosinor(x, params):
    r'''1-harmonic cosine function'''

    A = params['Amplitude']
    phi = params['Acrophase']
    T = params['Period']
    M = params['Mesor']

    return M + A*np.cos(2*np.pi/T*x+phi)


def _residual(params, x, data, fit_func):
    r'''Residual function to minimize'''

    model = fit_func(x, params)
    return (data-model)


class Cosinor():
    """
    Class for Cosinor analysis.

    Cornelissen, G. (2014). Cosinor-based rhythmometry.
    Theoretical Biology and Medical Modelling, 11(1), 16.
    https://doi.org/10.1186/1742-4682-11-16

    """

    def __init__(
        self,
        fit_params=None
    ):

        self.__fit_func = _cosinor  # Fit function
        self.__fit_obj_func = _residual

        if fit_params is None:
            fit_params = Parameters()
            # Default parameters for the cosinor fit function
            fit_params.add('Amplitude', value=50, min=0)
            fit_params.add('Acrophase', value=np.pi, min=0, max=2*np.pi)
            fit_params.add('Period', value=1440, min=0)  # Dummy value
            fit_params.add('Mesor', value=50, min=0)
        self.__fit_initial_params = fit_params

    @property
    def fit_func(self):
        r'''Cosinor fit function'''
        return self.__fit_func

    @property
    def fit_initial_params(self):
        r'''Initial parameters of the cosinor fit function'''
        return self.__fit_initial_params

    @fit_initial_params.setter
    def fit_initial_params(self, params):
        self.__fit_initial_params = params

    def fit(self,
            raw,
            params=None,
            method='leastsq',
            nan_policy='raise',
            reduce_fcn=None,
            verbose=False):
        """Fit the actigraphy data using a cosinor function.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be fitted.
        params: instance of Parameters [1]_, optional.
            Initial fit parameters. If None, use the default parameters.
            Default is None.
        method: str, optional
            Name of the fitting method to use [1]_.
            Default is 'leastsq'.
        nan_policy: str, optional
            Specifies action if the objective function returns NaN values.
            One of:

            * 'raise': a ValueError is raised
            * 'propagate': the values returned from userfcn are un-altered
            * 'omit': non-finite values are filtered

            Default is 'raise'.
        reduce_fcn: str, optional
            Function to convert a residual array to a scalar value for the
            scalar minimizers. Optional values are:
            * None : sum of squares of residual
            * negentropy : neg entropy, using normal distribution
            * neglogcauchy: neg log likelihood, using Cauchy distribution
            Default is None.
        verbose: bool, optional
            If set to True, display fit informations.
            Default is False.

        Returns
        -------
        fit_results : MinimizerResult
            Fit results.

        References
        ----------

        .. [1] Non-Linear Least-Squares Minimization and Curve-Fitting for
               Python.
               https://lmfit.github.io/lmfit-py/index.html

        """

        # Define the x range by converting timestamps to indices, in order to
        # deal with time series with irregular index.
        x = ((raw.data.index - raw.data.index[0])/raw.frequency).values

        # Minimize residuals
        fit_results = minimize(
            self.__fit_obj_func,
            self.fit_initial_params if params is None else params,
            method=method,
            args=(x,  raw.data.values, self.fit_func),
            nan_policy=nan_policy,
            reduce_fcn=reduce_fcn
        )
        # Print fit parameters if verbose
        if verbose:
            print(fit_report(fit_results))

        return fit_results

    def best_fit(self, raw, params):
        """Best fit function of the data.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be fitted.
        params: instance of Parameters [1]_
            Best fit parameters.

        Returns
        -------
        bestfit_data : pandas.Series
            Time series of the best fit data.

        References
        ----------

        .. [1] Non-Linear Least-Squares Minimization and Curve-Fitting for
               Python.
               https://lmfit.github.io/lmfit-py/index.html

        """

        # Define the x range by converting timestamps to indices, in order to
        # deal with time series with irregular index.
        x = ((raw.data.index - raw.data.index[0])/raw.frequency).values
        y = self.fit_func(x, params)

        return pd.Series(index=raw.data.index, data=y)

    def fit_reader(
        self,
        reader,
        params=None,
        method='leastsq',
        nan_policy='raise',
        reduce_fcn=None,
        verbose_fit=False,
        n_jobs=1,
        prefer=None,
        verbose_parallel=0
    ):
        """Fit the actigraphy data contained in a reader using a cosinor
        function.

        Parameters
        ----------
        reader : instance of RawReader
            Raw measurements to be fitted.
        params: instance of Parameters [1]_, optional.
            Initial fit parameters. If None, use the default parameters.
            Default is None.
        method: str, optional
            Name of the fitting method to use [1]_.
            Default is 'leastsq'.
        nan_policy: str, optional
            Specifies action if the objective function returns NaN values.
            One of:

            * 'raise': a ValueError is raised
            * 'propagate': the values returned from userfcn are un-altered
            * 'omit': non-finite values are filtered

            Default is 'raise'.
        reduce_fcn: str, optional
            Function to convert a residual array to a scalar value for the
            scalar minimizers. Optional values are:

            * None : sum of squares of residual
            * negentropy : neg entropy, using normal distribution
            * neglogcauchy: neg log likelihood, using Cauchy distribution

            Default is None.
        verbose_fit : bool.
            If True, print the fit summary.
            Default is False.
        n_jobs: int
            Number of CPU to use for parallel fitting
        prefer: str
            Soft hint to choose the default backendself.
            Supported option:'processes', 'threads'.
            See joblib package documentation for more info.
            Default is None.
        verbose_parallel: int
            Display a progress meter if set to a value > 0.
            Default is 0.

        """
        # avoid Parallel overhead if n_job == 1
        if(n_jobs == 1):
            results = [
                (
                    raw.name,
                    self.fit(
                        raw=raw,
                        params=params,
                        method=method,
                        nan_policy=nan_policy,
                        reduce_fcn=reduce_fcn,
                        verbose=verbose_fit
                    )
                ) for raw in reader.readers
            ]

        else:
            def parallel_fitting(
                raw, params, method, nan_policy, reduce_fcn, verbose
            ):
                fit_result = self.fit(
                    raw=raw,
                    params=params,
                    method=method,
                    nan_policy=nan_policy,
                    reduce_fcn=reduce_fcn,
                    verbose=verbose
                )
                fit_params = fit_result.params.valuesdict()
                # Add Goodness-of-fit informations
                fit_params['BIC'] = fit_result.bic
                fit_params['RedChiSq'] = fit_result.redchi
                return (
                    raw.name,
                    fit_params
                )
            results = Parallel(
                n_jobs=n_jobs,
                prefer=prefer,
                verbose=verbose_parallel
            )(
                delayed(parallel_fitting)(
                    raw=raw,
                    params=params,
                    method=method,
                    nan_policy=nan_policy,
                    reduce_fcn=reduce_fcn,
                    verbose=verbose_fit
                ) for raw in reader.readers
            )

        return pd.concat([
            pd.DataFrame(
                res[1],
                index=[res[0]]
            ) for res in results
        ], axis=0)
