import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # log(1 + exp(-b_i * A_i x)) = logsumexp([0, -b_i * A_i x])
        a = np.array([np.zeros(len(self.b)), -self.b * self.matvec_Ax(x)])
        return np.logsumexp(a, axis=0).mean() + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        # Градиент: -A^T(b * σ(-b * Ax)) / m + λx
        return -self.matvec_ATx(self.b * expit(-self.b * self.matvec_Ax(x))) / len(self.b) + self.regcoef * x

    def hess(self, x):
        # σ(-b_i * A_i x) * σ(b_i * A_i x)
        diag = expit(-self.b * self.matvec_Ax(x)) * expit(self.b * self.matvec_Ax(x))
        # Гессиан: A^T * diag(s) * A / m + λI
        return self.matmat_ATsA(diag) / len(self.b) + self.regcoef * np.eye(len(self.b))


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.Ax = None
        self.Ad = None
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        if self.Ax is None:
            self.Ax = self.matvec_Ax(x)
        if self.Ad is None:
            self.Ad = self.matvec_Ax(d)

        # log(1 + exp(-b_i * A_i(x + alpha * d)))
        a = np.logsumexp(np.array([np.zeros(len(self.b)), -self.b * (self.Ax + alpha * self.Ad)]), axis=0).mean()

        return a + 0.5 * self.regcoef * np.dot(x + alpha * d, x + alpha * d)

    def grad_directional(self, x, d, alpha):

        if self.Ax is None:
            self.Ax = self.matvec_Ax(x)
        if self.Ad is None:
            self.Ad = self.matvec_Ax(d)

        # ⟨-A^T(b * σ(-b * A(x+αd)))/m, d⟩ = -⟨b * σ(-b * A(x+αd)), Ad⟩/m
        data_derivative = -np.dot(self.b * expit(-self.b * (self.Ax + alpha * self.Ad)), self.Ad) / len(self.b)

        # ⟨λ(x + αd), d⟩
        reg_derivative = self.regcoef * np.dot(x + alpha * d, d)

        return data_derivative + reg_derivative


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x: A @ x
    matvec_ATx = lambda x: A.T @ x

    def matmat_ATsA(s):
        # '''
        # УЧИТЫВАЕМ SPARSE
        # '''
        # if scipy.sparse.isspmatrix_dia(A):
        #     return A.T @ (s[:, None] * A)
        return A.T @ s @ A

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
    grad = np.zeros_like(x)
    f_x = func(x)

    for i in range(x.size):
        dx = np.zeros_like(x)
        dx.flat[i] = eps
        grad.flat[i] = (func(x + dx) - f_x) / eps

    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """

    n = x.size
    hess = np.zeros((n, n))
    f_x = func(x)

    # f(x + eps * e_i) for i
    f_plus_i = np.zeros(n)
    for i in range(n):
        dx = np.zeros_like(x)
        dx.flat[i] = eps
        f_plus_i[i] = func(x + dx)

    for i in range(n):
        dx_i = np.zeros_like(x)
        dx_i.flat[i] = eps

        for j in range(i, n):
            dx_j = np.zeros_like(x)
            dx_j.flat[j] = eps

            f_plus_ij = func(x + dx_i + dx_j)
            hess_ij = (f_plus_ij - f_plus_i[i] - f_plus_i[j] + f_x) / (eps**2)

            hess[i, j] = hess_ij
            if i != j:
                hess[j, i] = hess_ij

    return hess

