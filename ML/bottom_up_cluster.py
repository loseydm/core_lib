import tqdm
import itertools as it

from math import inf
from numba import njit
from numbers import Number
from collections import namedtuple
from heapq import heappop, heappush
from typing import Iterable, Any, Tuple, List, Callable

Cluster = namedtuple('Cluster', ['parent_one', 'parent_two', 'members'])

@njit
def euclidean_distance(a: Iterable[Number], b: Iterable[Number]) -> Number:
  n, total = len(a), 0
  for i in range(n):
    total += (a[i] - b[i]) ** 2

  return total

class BottomUpCluster:
  """Clusters using a bottom-up approach, you can query any number of clusters once the tree is built"""

  def __init__(self, xs: Iterable[Any], distance: Callable[[Cluster, Cluster], Number], kind='minimum'):
    n = len(xs)
    c = 2 * n - 1  # Total number of clusters that will be produced

    self.distance, self.kind = distance, kind

    self.clusters, self.merged = [None] * c,  [False] * c
    self.cluster_distance = [[None] * c for _ in range(c)]

    c -= 1
    for i, x in enumerate(tqdm.tqdm(xs)):
      self.clusters[i] = Cluster(None, None, [i])

      for j in range(i, n):
        self.cluster_distance[i][j] = distance(xs[i], xs[j])
        self.cluster_distance[j][i] = self.cluster_distance[i][j]

    self.n, self.c = n, c+1
    self.fit()

  def __repr__(self):
    return 'BottomUpCluster(distance={}, kind={})'.format(self.distance, self.kind)

  def get_clusters(self, n):
    """Returns the n best clusters"""

    leaves = list()
    search_q = [-(self.c - 1)]
    while len(search_q) + len(leaves) != n:
      clust = -heappop(search_q)

      if (one := self.clusters[clust].parent_one) >= self.n:
        heappush(search_q, -one)
      else:
        leaves.append(self.clusters[one])

      if (two := self.clusters[clust].parent_two) >= self.n:
        heappush(search_q, -two)
      else:
        leaves.append(self.clusters[two])

    return [self.clusters[-i] for i in search_q] + leaves

  def fit(self) -> List[Cluster]:
    for id_number in tqdm.trange(self.n, self.c):
      min_i, min_j = BottomUpCluster.minimum_pair(self.n, self.merged, self.cluster_distance, id_number)

      self.merged[min_i], self.merged[min_j] = True, True
      self.clusters[id_number] = Cluster(min_i, min_j, self.clusters[min_i].members + self.clusters[min_j].members)

      for i in range(id_number):
        if not self.merged[i]:
          self.cluster_distance[i][id_number] = min(self.cluster_distance[min_i][i],
                                                    self.cluster_distance[min_j][i])

          self.cluster_distance[id_number][i] = self.cluster_distance[i][id_number]

  @njit
  @staticmethod
  def minimum_pair(n, merged, cluster_distance, max_id):
    min_dist = inf
    min_i, min_j = 0, 0
    for i in range(max_id):
      if not merged[i]:
        for j in range(i+1, max_id):
          if not merged[j] and cluster_distance[i][j] < min_dist:
            min_dist, min_i, min_j = cluster_distance[i][j], i, j

    return min_i, min_j

if __name__ == '__main__':
  inputs = [[-14,-5],[13,13],[20,23],[-19,-11],[-9,-16],[21,27],[-49,15],[26,13],[-46,5],[-34,-1],[11,15],[-49,0],[-22,-16],[19,28],[-12,-8],[-13,-19],[-41,8],[-11,-6],[-25,-9],[-18,-3]]
