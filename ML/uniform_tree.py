import sys
import numpy as np
import pandas as pd
import itertools as it
from random import shuffle
from statistics import mean
from dataclasses import dataclass
from typing import Any, Iterable, Tuple
from matplotlib.cm import gist_ncar as cm
from collections import namedtuple, deque
from string import ascii_lowercase as ABC

def to_hex(color: Tuple[float, float, float]) -> str:
	color = (color * 255).astype('uint8')
	return f'#{color[0]:02X}{color[1]:02X}{color[2]:02X}'

def format_question(split: str) -> str:
	if len(split) > 1:
		return f'Do they have {split}?'
	elif split.isupper():
		return f'Does their name start with {split}?'
	return f'Does their name contain {split}?'

def DFS_to_graphviz(tree: 'UniformTree', file: str, end_attribute: str):
	colors = cm(np.linspace(0, 1, 23))
	#shuffle(colors)
	colors = map(to_hex, colors)

	left_edge = '{} -> {} [labeldistance=2.5, labelangle=45, headlabel="Yes"]\n' 
	right_edge = '{} -> {} [labeldistance=2.5, labelangle=-45, headlabel="No"]\n'
	
	node_ = '{} [label="{}", fillcolor="{}"]\n'

	white = '#FFFFFF'

	file.write('digraph Tree {\n')
	file.write('node [shape=box, style="filled", color="black"] ;\n')
	file.write(node_.format(0, format_question(tree.root.split.replace('_', ' ').lower()), white))

	i, search_q = 1, [(0, tree.root)]
	while search_q:
		n, node = search_q.pop()

		left, right = node.left, node.right

		if left and left.split == end_attribute:
			i += 1
			file.write(node_.format(i, ', '.join(left.guess), next(colors)))
			file.write(left_edge.format(n, i))
		
		elif left:
			i += 1
			file.write(node_.format(i, format_question(left.split.replace('_', ' ').lower()), white))
			search_q.append((i, left))
			file.write(left_edge.format(n, i))

		if right and right.split == end_attribute:
			i += 1
			file.write(node_.format(i, ', '.join(right.guess), next(colors)))
			file.write(right_edge.format(n, i))

		elif right:
			i += 1

			file.write(node_.format(i, format_question(right.split.replace('_', ' ').lower()), white))
			search_q.append((i, right))
			file.write(right_edge.format(n, i))

	file.write('}')

class UniformTree:
	@dataclass
	class node:
		left: 'node'
		right: 'node'
		split: str
		guess: Any
	
	def __init__(self, data: Iterable[Any], attributes: Iterable[str], label_attribute: str):
		self.root = UniformTree.make_tree(data, attributes, label_attribute)

	@staticmethod
	def equity(data: Iterable[Any], attribute: Iterable[str]) -> float:
		"""Returns a float ∈ [0, .5] where 0 is a perfect 50/50 split 
		   and .5 is a redundant question"""

		return abs(mean(getattr(x, attribute) for x in data) - .5)

	@staticmethod
	def make_tree(data: Iterable[Any], attributes: Iterable[str], label_attribute: str) -> 'node':
		if not attributes or len(data) == 1:
			return UniformTree.node(None, None, label_attribute, tuple(getattr(x, label_attribute) for x in data))

		best_split = min(attributes, key=lambda attr: UniformTree.equity(data, attr))

		attributes = attributes.copy()
		attributes.remove(best_split)

		left, right = list(), list()
		for x in data:
			if getattr(x, best_split):
				left.append(x)
			else:
				right.append(x)

		left_node = UniformTree.make_tree(left, attributes, label_attribute)
		right_node = UniformTree.make_tree(right, attributes, label_attribute)

		return UniformTree.node(left_node, right_node, best_split, None)

if __name__ == '__main__':
    # Example usage
	players = pd.DataFrame(
		columns = ['Name', 'Male', 'A_hat', 'Glasses', 'Eye_color', 'Any_facial_hair', 'Hair_color', 'Earrings', 'A_receding_hairline'],
		
		data = 
		[
		['Philip', 	True, 	False, 	False, 	'Brown', 	True, 	'Black', 	False, 	False],
		['David', 	True, 	False, 	False, 	'Brown', 	True, 	'Blond', 	False, 	False],
		['Sam', 	True, 	False, 	True, 	'Brown', 	False, 	'White', 	False, 	True],
		['Maria', 	False, 	True, 	False, 	'Brown', 	False, 	'Brown', 	False, 	False],
		['Peter', 	True, 	False, 	False, 	'Blue', 	False, 	'White', 	False, 	False],
		['Bernard', True, 	True, 	False, 	'Brown', 	False, 	'Brown', 	False, 	False],
		['Frans', 	True, 	False, 	False, 	'Brown', 	False, 	'Red', 		False, 	False],
		['Tom', 	True, 	False, 	True, 	'Blue', 	False, 	'Black', 	False, 	True],
		['Anita', 	False, 	False, 	False, 	'Blue', 	False, 	'White', 	False, 	False],
		['Bill', 	True, 	False, 	False, 	'Brown', 	True, 	'Red', 		False, 	True],
		['Claire', 	False, 	True, 	True, 	'Brown', 	False, 	'Red', 		False, 	False],
		['Charles', True, 	False,	False,  'Brown', 	True, 	'Blond', 	False, 	False],
		['Richard', True, 	False, 	False, 	'Brown', 	True, 	'Brown', 	False, 	True],
		['George', 	True, 	True, 	False, 	'Brown', 	False, 	'White', 	False, 	False],
		['Joe', 	True, 	False, 	True, 	'Brown', 	False,  'Blond', 	False, 	False],
		['Paul', 	True, 	False, 	True, 	'Brown', 	False, 	'White', 	False, 	False],
		['Herman', 	True, 	False, 	False, 	'Brown', 	False, 	'Brown', 	False, 	True],
		['Alex', 	True, 	False, 	False, 	'Brown', 	True, 	'Black', 	False, 	False],
		['Susan', 	False, 	False, 	True, 	'Brown', 	False, 	'Blond', 	True, 	False],
		['Alfred', 	True, 	False, 	False,	'Blue', 	True, 	'Red', 		False, 	False],
		['Robert', 	True, 	False, 	False, 	'Blue', 	False, 	'Brown', 	False, 	False],
		['Anne', 	False, 	False, 	False, 	'Brown', 	False, 	'Black', 	True, 	False],
		['Eric', 	True, 	True, 	False, 	'Brown', 	False, 	'Blond', 	False, 	False],
		['Max', 	True, 	False, 	False, 	'Brown', 	True, 	'Black', 	False, 	False]
		]
	)

	players = players[players['Name'] != input('Who do I have? ')]
	print()

	players['An_Accessory'] = players['A_hat'] | players['Glasses'] | players['Earrings']
	players['A_hat_or_glasses'] = players['A_hat'] | players['Glasses']
	players['A_hat_or_earrings'] = players['A_hat'] | players['Earrings']
	players['Earrings_or_glasses'] = players['Earrings'] | players['Glasses']
	players['A_hat_and_glasses'] = players['A_hat'] & players['Glasses']
	players['A_hat_and_earrings'] = players['A_hat'] & players['Earrings']
	players['Earrings_and_glasses'] = players['Earrings'] & players['Glasses']

	for hair_color in players['Hair_color'].unique():
		players['{}_hair'.format(hair_color)] = players['Hair_color'] == hair_color

	for eye_color in players['Eye_color'].unique():
		players['{}_eyes'.format(eye_color)] = players['Eye_color'] == eye_color

	players.drop(['Eye_color', 'Hair_color'], axis=1, inplace=True)
	
	for letter in ABC:
		players[letter] = players['Name'].str.lower().str.contains(letter)
		players[letter] = players['Name'].str.contains(letter.upper())

	#attributes = players.columns[1:]
	#for one, two in it.combinations(attributes, 2):
	#	print(one, two)
	#	players['{}_and_{}'.format(one, two)] = players[one] & players[two]
	#	players['{}_or_{}'.format(one, two)] = players[one] | players[two]

	attributes = players.columns
	Player = namedtuple('Player', attributes)
	
	data = [Player(**row) for _, row in players.iterrows()]

	tree = UniformTree(data, set(attributes[1:]), 'Name')

	with open('UniformTree.dot', 'w') as UT:
		DFS_to_graphviz(tree, UT, 'Name')

	print('Dot file written!', end='\n\n')

	node, not_guessed = tree.root, None
	while True:
		split = node.split.replace('_', ' ').lower()

		T, F = node.left, node.right

		if node.split == 'Name':
			for name in node.guess:
				print(f'Is it {name}?')
				while (choice := input('Yes or no: ').lower()) not in ('yes', 'no'): pass
				if choice == 'yes':
					print('I win!')
					sys.exit()
		elif T.split == 'Name' or F.split == 'Name':
			if T.split == 'Name':
				check = T
				not_guessed = F
			else:
				check = F
				not_guessed = T
			for name in check.guess:
				print(f'Is it {name}?')
				while (choice := input('Yes or no: ').lower()) not in ('yes', 'no'):
					pass
				if choice == 'yes':
					print('I win!')
					sys.exit()
		else: print(format_question(node.split.replace('_', ' ')))
		while not_guessed is None and (choice := input('Yes or no: ').lower()) not in ('yes', 'no'): pass
		print()
		if not_guessed is not None:
			node = not_guessed
		elif choice == 'yes': node = T
		else: node = F
		not_guessed = None

