import Board
from Board import Chess
import chess.engine
import numpy as np

class StockfishData():
    
    def __init__(self, hash=512, time=0.01):
        self.stockfish = chess.engine.SimpleEngine.popen_uci("data/sf/bin/stockfish")
        self.stockfish.configure({"Threads": 1, "Hash": hash})
        self.limit = chess.engine.Limit(time=time)

    def get(self, size=32):
        n_moves = [int(abs(x)) for x in np.random.normal(Board.MOVE_LIMIT / 2.5, Board.MOVE_LIMIT / 9, size)]
        data = []
        for _ in range(size):
            chess_board = Chess()
            for move in range(n_moves.pop()):
                if chess_board.board.is_game_over():
                    chess_board.board.pop()
                    break
                chess_board.board.push(np.random.choice(list(chess_board.board.legal_moves)))
            state = chess_board.get_state()
            act = np.zeros(Board.ACTION_SPACE)
            move = self.stockfish.play(chess_board.board, self.limit).move
            if move:
                act[Board.ACTION_TO_INT[str(move)]] = 1
            val = 0.    # val = self.stockfish.analyse(chess_board.board, self.limit)["score"].relative.score(mate_score=100) / 10000.
            data.append((state, act, val))
        return data