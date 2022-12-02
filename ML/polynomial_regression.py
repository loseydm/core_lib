import random

from numbers import Number
from typing import Any, Iterable
from scratch.statistics import mean
from scratch.linear_algebra import Vector
from scratch.optimization import gradient_descent

class PolynomialRegression:
  """Polynomial regression model to one independent variable and one dependent variable"""

  def __init__(self, degrees: int, xs: Iterable[Any], ys: Iterable[Number], learning_rate: float, **kwargs):
    """Fits a polynomial regression model to xs and ys that minimizes the sum of squares error, kwargs go to sratch.optimization.gradient_descent"""

    if type(degrees) != int or degrees < 1:
      raise ValueError('degrees must be a positive integer')

    gradient_func = PolynomialRegression.gradient(xs, ys)

    betas = Vector([random.random() for _ in range(degrees)])

    self.degrees, self.learning_rate = degrees, learning_rate
    self.betas, converged, _ = gradient_descent(betas, gradient_func, learning_rate, **kwargs)

    if not converged:
      raise RuntimeError("Failed to converge, consider altering gradient descent's learning_rate or max_iterations")

  def __repr__(self):
    return 'PolynomialRegression(degrees={}, xs=[...], ys=[...], learning_rate={})'.format(self.degrees, self.learning_rate)

  def __str__(self):
    return ' + '.join('{:.2f}x^{}'.format(self.betas[i], i) if i > 0 else '{:.2f}'.format(self.betas[i]) for i in range(len(self.betas)))

  def predict(self, x):
    return sum(self.betas[i] * x ** i for i in range(self.degrees))

  def rsquared(self, xs, ys):
    n = len(xs)
    y_bar, y_hats = mean(ys), (self.predict(x) for x in xs)

    return 1.0 - sum((y - y_hat) ** 2 for y, y_hat in zip(ys, y_hats)) / sum((y - y_bar)**2 for y in ys)

  @staticmethod
  def gradient(xs: Iterable[Any], ys: Iterable[Number]):
    n = len(xs)

    def f(betas: Vector) -> Vector:
      m = len(betas)

      gradient = Vector.full(m, 0)
      for i in range(n):
        inner = ys[i] - sum(betas[j] * xs[i] ** j for j in range(m))

        for j in range(m):
          gradient[j] += inner * xs[i] ** j

      return -2 * gradient

    return f
