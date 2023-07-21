import os
import pickle as pkl
import numpy as np
import torch

C = np.sqrt(2)
EPSILON = 0.25
ALPHA = 0.03
class Node:
    
    def __init__(self, game, parent=None, action=None, prior=0):
        self.game = game
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value = 0.0

    def __len__(self):
        size = 1
        for child in self.children:
            size += len(child)
        return size

    def is_fully_expanded(self):
        return len(self.children) == self.game.get_legal_moves().sum()
    
    def select(self):
        
        def upper_confidence(node):
            if node.visits == 0:
                return float("inf")
            exploitation = node.value / node.visits
            exploration = C * np.sqrt(np.log(self.visits) / node.visits) * node.prior
            return exploitation + exploration

        node = self
        while node.is_fully_expanded() and node.children:
            node = max(node.children, key=upper_confidence)
        return node
    
    def expand(self, policy, dirichlet=False):
        nonzero = np.nonzero(policy)
        noise = np.random.dirichlet([ALPHA] * len(nonzero[0]))
        if dirichlet:
            for i in range(len(nonzero[0])):
                policy[nonzero[0][i]] = (1 - EPSILON) * policy[nonzero[0][i]] + EPSILON * noise[i]
        action = np.random.choice(len(policy), p=policy)
        child = Node(self.game.push(action), self, action, policy[action])
        self.children.append(child)
        return child

    def backpropagate(self, delta):
        self.value += delta
        self.visits += 1
        
        if self.parent is not None:
            self.parent.backpropagate(-delta)
        
    @staticmethod  
    def back(lst, value):
        arr = np.array(lst)
        value_arr = np.full(len(arr), value)
        if len(arr) % 2 == 0:
            mask = np.arange(len(arr)) % 2 != 0
        else:
            mask = np.arange(len(arr)) % 2 == 0
        mask = np.where(mask, 1, -1)
        arr[:-1] -= value_arr[:-1] * mask[1:]
        return arr.tolist()


class MCTS:
    
    def __init__(self, model):
        self.model = model
    
    @torch.no_grad()
    def search(self, search_node, num_searches=100, dirichlet=False):
        
        for _ in range(num_searches):
            node = search_node.select()
            
            if not node.parent:
                is_end = False
            else:
                value, is_end = node.parent.game.get_value(node.action)
            
            if not is_end:
                pol, value = self.model.evaluate(node.game)
                node = node.expand(pol, dirichlet)
                
            node.backpropagate(value)
        
        return self.get_action_distribution(search_node)
            
    def get_action_distribution(self, search_node):
        action_distribution = np.zeros(search_node.game.get_action_space())
        for child in search_node.children:
            action_distribution[child.action] = child.visits
        return action_distribution / np.sum(action_distribution)
    
    def save_tree(self, root, path="data/tree.txt"):
        with open(path, 'wb') as file:
            pkl.dump(root, file)
            
    def load_tree(self, path="data/tree.txt"):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pkl.load(file)
        else:
            return Node(self.model.game())