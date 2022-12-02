# After reading about KMeans, I implemented it as my first foray into machine learning.
# That experience ignited an ML passion that led to research, classes, and grad school applications, so
# this code holds sentimental value. It also just so happens to be < 100 lines! I implemented the algorithm
# using NumPy and make use of sklearn and matplotlib for an MNIST visualization.

import numpy as np

from typing import Union
from math import inf, ceil

from sklearn.datasets import fetch_openml

class KMeans:
	"""KMeans clustering implemented with NumPy"""

	def __init__(self, n_clusters: int, maximum_iterations: Union[int, None] = inf, verbose: bool = False):
		"""n_clusters: number of cluster k-means will attempt make, some could disappear
		   maximum_iterations: number of iterations k-means will perform, inf implies until convergence
		   verbose: bool indicating whether to show diagnostic messages during fitting"""

		self.n_clusters, self.maximum_iterations, self.verbose = n_clusters, maximum_iterations, verbose

	def __repr__(self):
		return f'KMeans(n_clusters = {self.n_clusters}, maximum_iterations = {self.maximum_iterations})'

	def fit(self, x: np.ndarray):
		"""Runs KMeans on 2d data where rows are samples and columns are features, uses paramaters set in constructor"""

		m, n = x.shape
		labels = KMeans._initial_labels(m, self.n_clusters)
		centroids = np.empty((self.n_clusters, n))

		# Allocate only once because of m x k x d size
		dist_matrix = np.empty((m, self.n_clusters, n))

		prev_labels = np.zeros_like(labels)
		
		i = 0
		while i < self.maximum_iterations and np.any(prev_labels != labels):
			if self.verbose:
				print(f'Iteration {i}')

			KMeans._update_centroids(x, centroids, labels)
			
			prev_labels, labels = labels, prev_labels
			
			KMeans._update_labels(x, centroids, labels, dist_matrix)

			i += 1

		return labels, centroids

	@staticmethod
	def _initial_labels(n_labels: int, k: int):
		"""Gives n labels sampled as evenly as possible"""
		
		labels = np.tile(np.arange(k), ceil(n_labels / k))[:n_labels]
		np.random.shuffle(labels)

		return labels
	
	@staticmethod
	def _update_centroids(x: np.ndarray, centroids: np.ndarray, labels: np.ndarray):
		"""Creates centroid group representatives given labels"""

		group_counts = np.c_[np.bincount(labels)]
		labels = labels[:, np.newaxis]

		for i in range(group_counts.size):
			np.sum(x, axis=0, out=centroids[i], where=(labels == i))

		np.divide(centroids, group_counts, out=centroids)

	@staticmethod
	def _update_labels(x: np.ndarray, centroids: np.ndarray, labels: np.ndarray, intermediate: np.ndarray):
		"""Updates label for each row to the closest centroid according to the l2 norm"""

		np.subtract(x[:, np.newaxis, :], centroids, out=intermediate)
		dist = np.linalg.norm(intermediate, axis=-1)

		np.argmin(dist, axis=-1, out=labels) 

if __name__ == '__main__':
	pixels, labels = fetch_openml('mnist_784', version=1, return_X_y=True)

	clustering = KMeans(10, verbose=True)
	labels, clusters = clustering.fit(pixels)

	# Reshape clusters into image form and show using grayscale
	plt.imshow(np.hstack(clusters.reshape(-1, 28, 28)), cmap='gray')
	plt.show()
