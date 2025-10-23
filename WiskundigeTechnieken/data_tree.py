# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
(c) 2025 Hogeschool Utrecht

Auteurs:
- Rianne van Os (rianne.vanos@hu.nl)
- Tijmen Muller (tijmen.muller@hu.nl)
"""

import numpy as np

class Node:
    def __init__(self, data=[], treshold=None, left_child=None, right_child=None):
        self.data = data
        self.treshold = treshold
        self.left_child = left_child
        self.right_child = right_child

        self.is_leaf = left_child is None and right_child is None
        
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf: data={self.data}"
        else:
            return f"Node: treshold={self.treshold:.2f}; data={self.data}"


class Tree:
    def __init__(self, root=None, max_depth=3):
        self.max_depth = max_depth
        self.root = root
        if root == None:
            self.root = Node()

    def __repr__(self):
        return self._repr_recursive(self.root, depth=0)

    def _build_recursive(self, data, depth):
        # TODO: implement
        return Node()

    def _repr_recursive(self, node, depth):
        # Base case: als een node een leaf is, dan tonen we simpelweg de
        # node (en hoeven we geen rekening te houden met verdere vertakkingen)
        if node.is_leaf:
            return f"{depth * '\t'}{node}"                                  
        
        # Recursion
        # TODO: implement
        return f"{depth * '\t'}{node}"  

    def _predict_recursive(self, node, n):
        # TODO: implement
        return node

    def build(self, data):
        self.root = self._build_recursive(data, depth=0)

    def predict(self, n):
        return self._predict_recursive(self.root, n)


if __name__ == "__main__":
    # Ter illustratie: een data tree handmatig
    # gebouwd met data = [2, 0, 1, 2]
    leaf0 = Node(np.array([0]))
    leaf1 = Node(np.array([1]))
    leaf2 = Node(np.array([2, 2]))
    node = Node(np.array([0, 1]), treshold=0.5, left_child=leaf0, right_child=leaf1)
    root = Node(np.array([2, 0, 1, 2]), treshold=1.25, left_child=node, right_child=leaf2)
    tree_simple = Tree(root)
    print(tree_simple)
