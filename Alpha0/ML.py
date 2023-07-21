from MonteCarlo import MCTS, Node
from Stockfish import StockfishData
from tqdm.notebook import trange
import numpy as np
import torch

ITERATIONS = 20
SELFPLAY_ITERATIONS = 100
PRETRAIN_ITERATIONS = 25000
SEARCHES = 200
EPOCHS = 1
BATCH_SIZE = 32

class Agent:

    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.mcts = MCTS(self.model)
        self.root = self.mcts.load_tree()
        self.mem = []
        
    def selfPlay(self):
        self.model.eval()
        for _ in trange(SELFPLAY_ITERATIONS):
            states = []
            actdists = []
            values = []
            
            node = self.root
            player = 1
            is_end = False
            while not is_end:
                state = node.game.get_state()
                action_distribution = self.mcts.search(node, SEARCHES, True)
                action = np.random.choice(len(action_distribution), p=action_distribution)
                value, is_end = node.game.get_value(action)
                
                for child in node.children:
                    if child.action == action:
                        node = child
                        break
                else:
                    raise Exception("unborn child")
                
                states.append(state)
                actdists.append(action_distribution)
                values.append(value*player)
                values = Node.back(values, value*player)
                
                player = (player-0.5)*2
                
            self.mem += list(zip(states, actdists, values))
        self.mcts.save_tree(self.root)
    
    def fit(self):
        self.model.train()
        for _ in range(EPOCHS):
            np.random.shuffle(self.mem)
            with trange(0, len(self.mem), BATCH_SIZE) as pbar:
                for batchIdx in pbar:
                    state, policy, value = map(np.array, zip(*self.mem[batchIdx:batchIdx + BATCH_SIZE]))
                    state, policy, value = [torch.tensor(x, dtype=torch.float32).to(self.model.device) for x in (state, policy, value)]
                    
                    policy_pred, value_pred = self.model(state)
                                    
                    policy_loss = torch.nn.functional.binary_cross_entropy(policy_pred, policy)
                    value_loss = torch.nn.functional.mse_loss(value_pred, value.unsqueeze(1))
                    loss = policy_loss + value_loss
                    pbar.set_description(f"Total: {loss.item():.4f}, Policy: {policy_loss.item():.4f}, Value: {value_loss.item():.4f}")
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        self.mem.clear()
        self.model.save()
    
    def pretrain(self):
        dataloader = StockfishData()
        for _ in trange(PRETRAIN_ITERATIONS//250):
            self.mem += dataloader.get(BATCH_SIZE*250)
            self.fit()
    
    def train(self):
        for _ in range(ITERATIONS):
            self.selfPlay()
            self.fit()
    
    def get_action(self, game):
        return self.mcts.search(Node(game))