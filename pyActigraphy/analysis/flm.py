import numpy as np
import pandas as pd
import statsmodels.api as sm
# from ..io.base import BaseRaw


class FLM():
    """ Class for Functional Linear Modelling"""

    def __init__(self, basis, sampling_freq, max_order):

        bases = ('fourier', 'ssa', 'wavelet')
        if basis not in bases:
            raise ValueError(
                '`basis` must be "%s". You passed: "%s"' %
                ('" or "'.join(bases), basis)
            )

        self.__basis = basis
        self.__sampling_freq = sampling_freq
        self.__max_order = max_order
        self.__basis_functions = None
        self.__beta = None

    def __create_basis_functions(self):

        phi = []
        # Construct the fourier functions (cosine and sine)
        if self.__basis == 'fourier':
            # self.sampling_freq = sampling_freq
            T = int(pd.Timedelta('24H')/pd.Timedelta(self.sampling_freq))
            omega = 2*np.pi / T
            t = np.linspace(0, T, T, endpoint=False)
            phi.append(np.cos(0 * t))
            for n in np.arange(1, self.max_order+1):
                phi.append(np.cos(n * omega * t))
                phi.append(np.sin(n * omega * t))

        self.basis_functions = phi

    def fit(self, raw, verbose=False):
        """Fit the actigraphy data using a basis function expansion.

        Parameters
        ----------
        raw : instance of BaseRaw or its child classes
            Raw measurements to be fitted.
        verbose : bool.
            If True, print the fit summary.
            Default is False.

        Returns
        -------
        y_est : ndarray
            Returns the functional form of the actigraphy data.
        """
        if self.basis_functions is None:
            print("Create first the basis functions: {}".format(self.__basis))
            self.__create_basis_functions()
        if self.basis_functions == []:
            print("Basis functions are empty. Returning None")
            return None

        X = np.stack(self.basis_functions, axis=1)
        y = raw.average_daily_activity(
            binarize=False,
            freq=self.sampling_freq
        ).values
        model = sm.OLS(y, X)
        results = model.fit()

        if verbose:
            print(results.summary())

        self.__beta = results.params
        y_est = np.dot(X, self.beta)
        return y_est

    @property
    def sampling_freq(self):
        """The sampling frequency of the basis functions."""
        return self.__sampling_freq

    @sampling_freq.setter
    def sampling_freq(self, value):
        self.__sampling_freq = value

    @property
    def max_order(self):
        """The maximal number of basis functions."""
        return self.__max_order

    @property
    def basis_functions(self):
        """The basis functions."""
        return self.__basis_functions

    @basis_functions.setter
    def basis_functions(self, value):
        self.__basis_functions = value

    @property
    def beta(self):
        """The scalar parameters of the basis function expansion."""
        return self.__beta
