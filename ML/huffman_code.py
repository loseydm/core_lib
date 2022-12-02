from typing import Iterable, Any, Dict, List
from heapq import heappush, heappop
from dataclasses import dataclass
from collections import Counter

class HuffmanCode:
	@dataclass
	class node:
		frequency: float
		left: 'node'
		right: 'node'
		character: Any

		def __lt__(self, other: 'node') -> bool:
			return self.frequency < other.frequency

	def __init__(self, items: Iterable[Any]):
		self.items = items

		p_queue = list()
		for item in (items := Counter(items)):
			p_queue.append(HuffmanCode.node(items[item], None, None, item))

		self.root = self.make_tree(p_queue)

	def __repr__(self):
		return f'HuffmanCode({self.items!r})'

	@staticmethod
	def make_tree(p_queue: List['node']) -> 'node':
		if len(p_queue) == 1:
			return p_queue.pop()

		one, two = heappop(p_queue), heappop(p_queue)
		
		heappush(p_queue,
			HuffmanCode.node(one.frequency + two.frequency, one, two, None))

		return HuffmanCode.make_tree(p_queue)

	def make_lookup_table(node: 'node', lookup_table: Dict[Any, str], prefix: str) -> Dict[Any, str]:
		if node.left is not None:
			HuffmanCode.make_lookup_table(node.left, lookup_table, prefix + '0')

		if node.right is not None:
			HuffmanCode.make_lookup_table(node.right, lookup_table, prefix + '1')

		if node.left is None and node.right is None:
			lookup_table[node.character] = prefix

		return lookup_table