import chess
import numpy as np
import torch
import matplotlib.pyplot as plt

MOVE_LIMIT = 100
with open('data/actions.txt', 'r') as f:
    ACTIONS = f.read().split(",")
INT_TO_ACTION = dict(enumerate(ACTIONS))
ACTION_TO_INT = {v: k for k, v in INT_TO_ACTION.items()}
ACTION_SPACE = len(ACTIONS) # AlphaZero has an action space of 4672 for comparison --> https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work
PIECE_TO_VAL = {"P":0.005, "N":0.015, "B":0.015, "R":0.025, "Q":0.045}
PIECE_TO_INT = {"P":1/6, "N":2/6, "B":3/6, "R":4/6, "Q":5/6, "K":6/6}
class Chess():

    def __init__(self, board=chess.Board()):
        self.board = board.copy()

    def get_state(self):
        # board + attack maps
        board_pieces = np.zeros(64)
        attacked_by_me = []
        attacked_by_you = []
        for pos in self.board.piece_map():
            
            piece = self.board.piece_at(pos)
            symbol = piece.symbol().upper()
            turn_piece = piece.color == self.board.turn
            val = PIECE_TO_INT[symbol] if turn_piece else -PIECE_TO_INT[symbol]

            col, row = pos % 8, pos // 8
            board_pieces[row * 8 + col] = val

            if turn_piece:
                attacked_by_me += list(self.board.attacks(pos))
            else:
                attacked_by_you += list(self.board.attacks(pos))

        attacked_by_me = np.bincount(attacked_by_me, minlength=64) / 4.
        attacked_by_you = np.bincount(attacked_by_you, minlength=64) / 4.

        # legal move map
        legals = np.zeros(64)
        for action in [str(i) for i in self.board.legal_moves]:
            move = chess.Move.from_uci(action)
            legals[move.to_square] += 0.2
            legals[move.from_square] -= 0.125

        full_board = np.concatenate([board_pieces, attacked_by_me, attacked_by_you, legals])
        return torch.from_numpy(np.flip(full_board.reshape(-1,8,8), axis=1).copy()).float()
    
    def get_legal_moves(self):
        legal_moves = np.zeros(ACTION_SPACE)
        np.put(legal_moves, [ACTION_TO_INT[str(i)] for i in self.board.legal_moves], 1)
        return legal_moves

    def push(self, action):
        chess_board = self.board.copy()
        chess_board.push_uci(INT_TO_ACTION[int(action)])
        return Chess(chess_board)
    
    def get_value(self, action):
        if len(self.board.move_stack) >= MOVE_LIMIT-1:
            return 0, True
        val, end = 0, False
        if action != None:
            if type(action) != str:
                action = INT_TO_ACTION[int(action)]
            move = chess.Move.from_uci(action)
            if len(action) == 5:
                val += PIECE_TO_VAL[action[-1].upper()]
            if self.board.is_castling(move):
                val += 0.01
            if self.board.is_capture(move):
                captured = self.board.piece_at(move.to_square)
                if captured:
                    val += PIECE_TO_VAL[captured.symbol().upper()]
                else: # en passant
                    val += 0.0075
            self.board.push(move)
            if self.board.is_checkmate():
                val += 0.5
                end = True
            elif self.board.is_check():
                val += 0.01
            elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_fifty_moves() or self.board.is_fivefold_repetition():
                end = True
            self.board.pop()
        return val, end
    
    def show(self, board=False):
        if board:
            display(self.board)
        else:
            arr = self.get_state()
            fig, axs = plt.subplots(1, arr.shape[0], figsize=(32, 16))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(arr[i], cmap='bone')
            plt.show()
      
    @classmethod  
    def get_action_space(cls):
        return ACTION_SPACE
    @classmethod  
    def int_to_action(cls, action):
        return INT_TO_ACTION[action]