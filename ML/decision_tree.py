import tqdm
import numpy as np

#from mnist import MNIST
from math import log, inf
from dataclasses import dataclass
from typing import List, Any, Dict, Hashable, Set
from collections import namedtuple, defaultdict, Counter
from scratch.statistics import bootstrap

Candidate = namedtuple('Candidate', ('level', 'language', 'has_twitter', 'has_phd', 'should_hire'))

inputs = [
          Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
         ]

def split_data(inputs: List[Any], attr: str) -> Dict[str, List[Any]]:
  """Splits the inputs into bins on the attr"""

  bins = defaultdict(list)
  for x in inputs:
    bins[getattr(x, attr)].append(x)

  return bins

def entropy(ps: List[Any]) -> List[float]:
  """Calculates the entropy of information"""

  return -sum(p * log(p, 2) for p in ps)

def class_probabilities(groups: Dict[Any, int]) -> List[float]:
  """Returns the probability of seeing each label, expect a dictionary or Counter"""

  n = sum(groups.values())
  return [groups[group] / n for group in groups]

def weighted_entropy(inputs: List[Any], attribute: str, label: str):
  """Calculates the weighted entropy for splitting on the given attribute"""

  n = len(inputs)

  groups = defaultdict(Counter)
  for datum in inputs:
    key, value = getattr(datum, attribute), getattr(datum, label)
    groups[key][value] += 1

  return sum( sum(group.values()) / n * entropy(class_probabilities(group)) for group in groups.values())

class Bagging:
  def __init__(self, inputs: List[Any], attributes: List[str], label_attribute: str, k: int, max_levels: int = inf, p: float = 1):
    self.trees = [DecisionTree(bootstrap(inputs, p), attributes, label_attribute, max_levels)
                    for _ in tqdm.trange(k)]

  def predict(self, value: Any):
    return Counter(tree.predict(value) for tree in self.trees).most_common(1)[0]


class DecisionTree:
  """A classification decision tree that accepts categorical variable only"""

  @dataclass
  class Node:
    attribute: str
    children: Dict[str, 'Node']
    prediction: Any

    def __repr__(self):
      if self.children is None:
        return 'Node({}, None, {})'.format(self.attribute, self.prediction)

      return 'Node({}, {}, {})'.format(self.attribute, self.children.keys(), self.prediction)

    def predict(self, observation):
      if self.children is None:
        return self.prediction

      try:
        return self.children[getattr(observation, self.attribute)].predict(observation)

      except KeyError:
        return self.prediction


  def __init__(self, inputs: List[Any], attributes: List[str], label_attribute: str, max_levels: int = inf):

    if max_levels < 0 or not isinstance(max_levels, int) and max_levels != inf:
      raise ValueError('max_levels must be a nonnegative integer or inf')

    self.root = DecisionTree.build_tree(inputs, set(attributes), label_attribute, max_levels)

  def __repr__(self):
    return 'DecisionTree(*args, **kwargs)'

  def predict(self, value: Any) -> Any:
    return self.root.predict(value)

  @staticmethod
  def build_tree(inputs: List[Any], attributes: Set[str], label_attribute: str, max_levels: int) -> 'Node':
    """Builds a greedy decision tree from the current attributes and inputs"""

    attributes = attributes.copy()

    # Most likely class based on simple count, all nodes need this in the case of an undefined attribute value
    counts = Counter(getattr(x, label_attribute) for x in inputs)
    prediction = counts.most_common(1)[0][0]

    if len(counts) == 1 or max_levels == 0:
      return DecisionTree.Node(None, None, prediction)

    attr = DecisionTree.best_split(inputs, attributes, label_attribute)

    attributes.remove(attr)

    groups = split_data(inputs, attr)
    children = {group: DecisionTree.build_tree(groups[group], attributes, label_attribute, max_levels - 1) for group in groups}

    return DecisionTree.Node(attr, children, prediction)

  @staticmethod
  def best_split(inputs: List[Any], attributes: Set[str], label_attribute: str):
    """Returns the best split at a given level on the given inputs"""

    return min(attributes, key=lambda a: weighted_entropy(inputs, a, label_attribute))

if __name__ == '__main__':
  def write_binary_image(img: np.ndarray, prediction: str, img_dir: str):
    i = 0
    while os.path.exists((path := '{}/terminal_node_predict_{}_number_{}.jpg'.format(img_dir, prediction, i))):
        i += 1

  def DFS_decision_tree(node: DecisionTree.Node, img: np.ndarray, f) -> None:
    if node.children is None:
        f(img, node.prediction)
        return

    i = int(node.attribute.lstrip('pixel_'))
    img[i] = 255

    for child in node.children:
        DFS_decision_tree(node.children[child], img, f)

    img[i] = 0

  mnist = MNIST()

  label = ('number',)
  predictors = tuple('pixel_{}'.format(n) for n in  range(784))
  MNIST_img = namedtuple('MNIST_img', label + predictors)

  train_images = [MNIST_img(mnist.train_set.labels[i].argmax(), *(mnist.train_set.images[i] != 0)) for i in range(60_000)]

  dt = DecisionTree(train_images, predictors, 'number', 5)

  test_images = [MNIST_img(mnist.test_set.labels[i].argmax(), *(mnist.test_set.images[i] != 0)) for i in range(10_000)]

  labels = Counter()
  for img in test_images:
    labels[(img.number, dt.predict(img))] += 1

