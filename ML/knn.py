import matplotlib.pyplot as plt

from dataclasses import dataclass
from collections import Counter, defaultdict
from random import shuffle
from typing import List
from dsfs import Vector

@dataclass
class Label:
    point: Vector
    label: str
    
def majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    
    winner, winner_count = vote_counts.most_common(1)[0]
    if sum(winner_count == vote_counts[x] for x in vote_counts) == 1:
        return winner
    
    return majority_vote(labels[:-1])

def knn_classify(k: int, labelled_points: List[Label], new_point: Vector) -> str:
    """Naive knn for labelled point vs new_point"""
    
    by_distance = sorted(labelled_points, key=lambda o: new_point.distance(o.point))
    
    labels = [point.label for point in by_distance[:k]]
    
    return majority_vote(labels)
    
def train_test_split(data, p):
    shuffle(data)
    
    i = int(len(data) * p)
    return data[:i], data[i:]
    
if __name__ == '__main__':
    import requests
    
    data = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    
    with open('iris.dat', 'w') as f:
        f.write(data.text)
    
    data = []
    with open('iris.dat') as d:
        for line in d:
            records = line.strip().split(',')
            records, label = list(map(float, records[:-1])), records[-1]
            data.append(Label(Vector(records), label))
    
    data.pop()
    
    labels = {d.label for d in data}
    
    cols = 'sepal_length, sepal_width, petal_length, petal_width'.split(', ')
    
    marks = ['+', '.', 'x'] 
    color = ['blue', 'orange', 'green']
    pairs = [(i, j) for i in range(len(cols)) for j in range(i+1, len(cols))]

    fig, ax = plt.subplots(2, 3)
    for i in range(2):
        for j in range(3):
            one, two = pairs[i * 3 + j]

            ax[i][j].set_title(f'{cols[one]} vs. {cols[two]}')
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

            for k, label in enumerate(labels):
                xs = [d.point.xs[one] for d in data if d.label == label]
                ys = [d.point.xs[two] for d in data if d.label == label]
                
                ax[i][j].scatter(xs, ys, marker=marks[k], label=label)
    
    fig.tight_layout()
    plt.show()

    train, test = train_test_split(data, .7)
    
    correct = 0
    confusion_matrix = defaultdict(int)
    for point in test:
        predicted = knn_classify(4, data, point.point)
        
        if predicted == point.label:
            correct += 1
        
        confusion_matrix[(point.label, predicted)] += 1
    
    print(confusion_matrix)
    print('{:.4f}% accuracy'.format(100 * correct / len(test)))
        
    