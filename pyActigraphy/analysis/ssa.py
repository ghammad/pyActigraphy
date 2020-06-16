# from functools import lru_cache
from numba import jit, prange  # float64, int32
import numpy as np
import pandas as pd
from scipy import linalg


@jit(nopython=True)
def _weights(L, K):

    N = L + K
    # weights = np.empty(N-1, dtype=np.float32)
    weights = np.empty(N-1, dtype=np.int32)
    for k in prange(1, L):
        weights[k-1] = k

    # for k in range(L,K+1):
    weights[L-1:K] = L

    for k in prange(K+1, N):
        weights[k-1] = N-k

    return weights


def _weighted_scalar_product(X, Y, w):
    return np.dot(X, np.multiply(Y, w).T)


def _weighted_correlation(X, Y, w):

    w_norm_X = np.sqrt(_weighted_scalar_product(X, X, w))
    w_norm_Y = np.sqrt(_weighted_scalar_product(Y, Y, w))

    w_rho = _weighted_scalar_product(X, Y, w) / (w_norm_X*w_norm_Y)

    return w_rho


@jit(nopython=True, parallel=True)
def _x_elementary(U, s, Vh, L, K, i):

    X_i = np.empty((L, K), dtype=np.float32)

    # Implement the dot product s * U[,i] x Vh[i].T
    sVh_i = s*Vh[i]
    for j in prange(L):
        X_i[j] = U[j, i]*sVh_i

    return X_i


@jit(nopython=True, parallel=True)
def _diagonal_averaging(X):

    L, K = X.shape
    L_star, K_star = min(L, K), max(L, K)
    # N_star = L_star + K_star
    if not L < K:
        X = X.T

    sum_antidiags = np.empty(L_star + K_star - 1, dtype=np.float32)
    for k in prange(1-L_star, K_star):
        # Avoid using np.flipud as it does not compile with numba.
        # Besides, it seems slower than [::-1,...]
        sum_antidiags[k+L_star-1] = np.trace(X[::-1, ...], offset=k)

    scale_factors = _weights(L_star, K_star)
    # scale_factors = np.empty(N_star-1, dtype=np.float32)
    # for k in prange(1, L_star):
    #     scale_factors[k-1] = k
    #
    # # for k in range(L_star,K_star+1):
    # scale_factors[L_star-1:K_star] = L_star
    #
    # for k in prange(K_star+1, N_star):
    #     scale_factors[k-1] = N_star-k

    sum_antidiags /= scale_factors

    return sum_antidiags


class SSA():
    """ Class for Singular Spectrum Analysis"""

    def __init__(self, data, window_length='24H'):

        self.__data = data
        self.__freq = pd.Timedelta(data.index.freq)
        self.__window_dim = int(pd.Timedelta(window_length)/self.__freq)
        self.__L = self.__window_dim
        self.__K = len(data.values) - self.__L + 1
        self.__U = None
        self.__sigma = None
        self.__Vh = None
        self.__lambda_s = None

    @property
    def window_dim(self):
        r"""Window dimension for immersion of the signal.
        Window dimension (or embedding dimension) in number of epochs.
        """
        return self.__window_dim

    # @lru_cache(maxsize=6)
    def trajectory_matrix(self):
        r'''Trajectory matrix of the signal.
        Time-series :math:`x=(x_0,x_1,\dots,x_n,\dots,x_{N−1})^\intercal`, with
        length N, represents the signal under analysis. The mapping of this
        signal into a matrix A, of dimension L × K , assuming :math:`L \leq K`,
        is called immersion, and can be defined as:

        .. math::

            A = \begin{bmatrix}
                x_{0} & x_{1} & x_{2} & \dots  & x_{K-1} \\
                x_{1} & x_{2} & x_{3} & \dots  & x_{K} \\
                \vdots & \vdots & \vdots & \ddots & \vdots \\
                x_{L-1} & x_{L} & x_{L+1} & \dots  & x_{N-1}
                \end{bmatrix}

        where L is the window length, or embedding dimension, and
        :math:`K = N − L + 1`. A is a Hankel matrix, called the trajectory
        matrix.
        '''
        ts = self.__data.values
        c = ts[:self.__L]
        r = ts[-self.__K:]
        A = linalg.hankel(c, r)
        return A

    def fit(self, check_finite=False, overwrite_a=True):
        r'''Singular value decomposition of the trajectory matrix.
        Wrapper around the scipy.linalg.svd function.

        Parameters
        ----------
        overwrite_a : bool, optional
            Whether to overwrite `a`; may improve performance.
            Default is False.
        check_finite : bool, optional
            Whether to check that the input matrix contains only finite
            numbers. Disabling may give a performance gain, but may result in
            problems (crashes, non-termination) if the inputs do contain
            infinities or NaNs.
            Default is True.


        Notes
        -----

        Factorization of the trajectory matrix A, using Singular Value
        Decomposition (SVD), yields to [1]_:

        .. math::

            A &= U\Sigma V^\intercal \\
              &= \sum_{r=1}^{R} \sigma_r u_r v_{r}^\intercal

        where :math:`R = rank(A) \leq L`, :math:`{u_1,\ldots, u_d }` is the
        corresponding orthonormal system of the eigenvectors of the matrix
        :math:`S = AA^{\intercal}` such as :math:`ui \cdot uj = 0` for
        :math:`i \neq j` and :math:`\lVert u_r \rVert = 1`,
        :math:`v_r = A^{\intercal} u_r / \sigma_r`, and :math:`\Sigma` is a
        diagonal matrix :math:`\in \mathbb{R}^{L×K}`, whose diagonal elements
        :math:`{\sigma_r}` are the singular values of A. The eigenvalues of
        :math:`AA^\intercal` are given by :math:`\lambda_r = \sigma_r^2`.


        References
        ----------

        .. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum
               Analysis for Time Series. Springer Berlin Heidelberg
               http://doi.org/10.1007/978-3-642-34913-3
        '''

        A = self.trajectory_matrix()
        U, s, Vh = linalg.svd(
            A,
            full_matrices=False,
            check_finite=check_finite,
            overwrite_a=overwrite_a
        )
        self.__U = U
        self.__sigma = np.diag(s)
        self.__Vh = Vh
        self.__lambda_s = np.square(s)/np.sum(np.square(s))

    @property
    def U(self):
        r'''Unitary matrix having left singular vectors as columns.
        '''
        return self.__U

    @property
    def sigma(self):
        r'''Singular values, sorted in decreasing order.
        '''
        return self.__sigma

    @property
    def Vh(self):
        r'''Unitary matrix having right singular vectors as rows.
        '''
        return self.__Vh

    @property
    def lambda_s(self):
        r'''Partial variances, sorted in decreasing order.
        The partial variances :math:`\lambda_k = \sigma^2_k`, ordered according
        to magnitude from the most to the least dominant, where
        :math:`\lambda_k` can be interpreted as the variance of the
        *sub phase-space* of time-series component :math:`g_k(n)` and where
        :math:`\lambda_{tot} = \sum^r_{k=1} \lambda_k` is the total variance of
        the phase space of the original time series.
        '''
        return self.__lambda_s

    def X_elementary(self, r):
        r'''Elementary matrix

        Parameters
        ----------
        r: int
            Index of the elementary matrix.
            Must lower or equal to the embedding dimension, L.

        Returns
        -------
        x_elem: ndarray of shape (L,K)


        Notes
        -----

        The SVD of the trajectory matrix X can be written as [1]_ :

        .. math:

            X = X_1 + \ldots + X_R

        where :math:`X_r = \sqrt{\lambda_r} u_r v_{r}^\intercal`.

        The matrices :math:`X_r` have rank 1. Such matrices are sometimes
        called *elementary* matrices.


        References
        ----------

        .. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum
               Analysis for Time Series. Springer Berlin Heidelberg
               http://doi.org/10.1007/978-3-642-34913-3
        '''
        #  TODO: check if r is in range

        X_r = _x_elementary(
            self.__U,
            self.__sigma[r][r],
            self.__Vh,
            self.__L,
            self.__K,
            r
        )

        return X_r

    def X_tilde(self, r):
        r'''Diagonal averaged matrix.

        Parameters
        ----------
        r: int or list of int
            Index of the elementary matrix to be diagonal-averaged.
            Must be lower than or equal to the embedding dimension, L.
            If a list of indices is given instead, the corresponding elementary
            matrices are grouped (ie. reduced to a single matrix by summation)
            before diagonal-averaging.

        Returns
        -------
        x_tilde: ndarray of shape (M,)


        Notes
        -----

        [1]_ : if the components of the series are separable and the indices
        are being split accordingly, then all the matrices in the expansion
        :math:`X = X_{I_1} + \ldots + X_{I_m}` are the Hankel matrices.
        We thus immediately obtain the decomposition
        :math:`x_n = \sum_{k=1}^m \tilde{x}_n^{(k)}` of the original series:
        for all k and n, :math:`\tilde{x}_n^{(k)}` is equal to all entries
        :math:`x^{(k)}_{ij}` along the antidiagonal
        :math:`{(i, j)| i + j = n+1}` of the matrix :math:`X_{Ik}`. In
        practice, however, this situation is not realistic. In the general
        case, no antidiagonal consists of equal elements. We thus need a formal
        procedure of transforming an arbitrary matrix into a Hankel matrix and
        therefore into a series. As such, we shall consider the procedure of
        *diagonal averaging*, which defines the values of the time series

        .. math::

            \tilde{\mathbb{X}}^{(k)} = \left(
                \tilde{x}^{(k)}_1, \ldots, \tilde{x}^{(k)}_N \right)

        as averages for the corresponding antidiagonals of the matrices
        :math:`X_{I_k}`.

        * for :math:`1 \leq n < L^{\star}`:

          .. math::
             \tilde{x}_n^{(k)} = \frac{1}{n} *
             \sum_{m=1}^{n} x^{\star}_{I_k, (m,n-m+1)}

        * for :math:`L^{\star} \leq n < K^{\star}`:

          .. math::
             \tilde{x}_n^{(k)} = \frac{1}{L^{\star}} *
             \sum_{m=1}^{L^{\star}} x^{\star}_{I_k, (m,n-m+1)}

        * for :math:`K^{\star} < n \leq N`:

          .. math::
             \tilde{x}_n^{(k)} = \frac{1}{N-n+1} *
             \sum_{m=n-K^{\star}+1}^{N-K^{\star}+1} x^{\star}_{I_k, (m,n-m+1)}

        References
        ----------

        .. [1] Golyandina, N., & Zhigljavsky, A. (2013). Singular Spectrum
               Analysis for Time Series. Springer Berlin Heidelberg
               http://doi.org/10.1007/978-3-642-34913-3
        '''
        if isinstance(r, list):
            X_elementaries = [self.X_elementary(i) for i in r]
            from functools import reduce
            X_elementary = reduce((lambda x, y: np.add(x, y)), X_elementaries)
        else:
            X_elementary = self.X_elementary(r)

        X_tilde = _diagonal_averaging(X_elementary)

        return X_tilde

    def reconstructed_signal(self, n):
        r'''Reconstructed signal from diagonal averaged matrices.

        Parameters
        ----------
        n: array of int
            Indices of the diagonal-averaged matrices to merge.
            Must be lower than or equal to the embedding dimension, L.

        Returns
        -------
        reco: pandas.Series

        '''

        X_tildes = [self.X_tilde(i) for i in n]

        # add the X_tilde matrices recursively
        from functools import reduce
        X_reco = reduce((lambda x, y: np.add(x, y)), X_tildes)

        reco_signal = pd.Series(
            data=X_reco,
            index=self.__data.index
        )

        return reco_signal

    def w_correlation_matrix(self, k):
        r'''W-correlation matrix.

        Parameters
        ----------
        n: int
            Maximal index of the diagonal-averaged matrices to use.
            Must be lower than or equal to the embedding dimension, L.

        Returns
        -------
        wmat: numpy.ndarray

        '''

        n = range(k)

        w_corr_mat = np.empty((k, k))

        w = _weights(self.__L, self.__K)

        X_tildes = [self.X_tilde(i) for i in n]

        for i in n:
            for j in n[i:]:
                w_corr = _weighted_correlation(
                    X_tildes[i],
                    X_tildes[j],
                    w
                )
                w_corr_mat[i][j] = w_corr
                w_corr_mat[j][i] = w_corr

        return w_corr_mat
