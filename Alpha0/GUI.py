import chess
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

class ChessGUI:

    def __init__(self):
        
        while True:
            self.side = input("Side (white/black): ").strip().lower()
            if self.side in ["white", "black"]:
                break
        
        self.root = tk.Tk()
        self.root.title("Chess")
        
        self.board = chess.Board()
        self.history = []
        self.images = self.load_piece_images()

        self.canvas = tk.Canvas(self.root, width=640, height=640)
        self.canvas.pack()
        self.textbox = tk.Text(self.root, height=3, width=80, font=("Arial", 15), state="disabled")
        self.textbox.pack()
        self.draw_board()
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Left>", self.on_back)
        self.root.bind("<Right>", self.on_forward)
        
        self.pos1 = None
        self.pos2 = None
        
        self.over = False
        self.root.mainloop()
    
    def load_piece_images(self):
        pieces = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']
        ims = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
        images = dict(zip(pieces, ims))
    
        for piece in images.keys():
            image = Image.open(f'data/pieces/{images[piece]}.png')
            image = image.resize((70, 70), Image.LANCZOS)
            images[piece] = ImageTk.PhotoImage(image)
        return images

    def board_to_array(self):
        board_array = np.zeros((8,8), dtype=str)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                board_array[i // 8][i % 8] = piece.symbol()
        if self.side == "white":
            return np.flip(board_array, axis=0)
        return board_array
    
    def draw_board(self):
        colors = ["white", "gray"]
        board_array = self.board_to_array()
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                self.canvas.create_rectangle(col * 80, row * 80, (col + 1) * 80, (row + 1) * 80, fill=color)
                piece = board_array[row][col]
                if piece:
                    self.canvas.create_image(col * 80 + 40, row * 80 + 40, image=self.images[piece])

        self.update_textbox(f"Turn: {'WHITE' if self.board.turn else 'BLACK'} | FEN: {self.board.fen()}")

    def get_event_pos(self, event):
        col = event.x // 80
        if self.side == "white":
            row = 7 - event.y // 80
        else:
            row = event.y // 80
        return chess.square_name(row * 8 + col)
    
    def on_click(self, event):
        if not self.over:
            self.pos1 = self.get_event_pos(event)
    
    def on_release(self, event):
        self.history.clear()
        if self.pos1:
            self.pos2 = self.get_event_pos(event)
            action = self.pos1 + self.pos2
            if action in list(str(x)[:4] for x in self.board.legal_moves):
                action = chess.Move.from_uci(action)

                if (self.pos2[-1] == "8" or self.pos2[-1] == "1") and self.board.piece_at(chess.parse_square(self.pos1)).symbol() in ['P', 'p']: # promotion
                    self.update_textbox("Promotion! What would you like to promote to?")
                    while True:
                        promotion = input("Promotion (Q/R/N/B): ").strip().upper()
                        if promotion in ["Q", "R", "N", "B"]:
                            break
                    promote_to = {"Q":chess.QUEEN, "R":chess.ROOK, "N":chess.KNIGHT, "B":chess.BISHOP}
                    action.promotion = promote_to[promotion]

                self.board.push(action)
                self.draw_board()
            self.pos1 = None
            self.pos2 = None
            self.check_mate()

            if not self.over and ((self.board.turn and self.side == "black") or (not self.board.turn and self.side == "white")):
                self.board.push_uci(self.get_action())
                self.draw_board()
            self.check_mate()
                
    def check_mate(self):
        if self.board.is_checkmate():
            self.update_textbox("Checkmate!")
        elif self.board.is_stalemate():
            self.update_textbox("Stalemate!")
        elif self.board.is_fifty_moves():
            self.update_textbox("50 move repetition!")
        elif self.board.is_fivefold_repetition():
            self.update_textbox("5 fold repetition!")
        elif self.board.is_insufficient_material():
            self.update_textbox("Insufficient material!")
        if self.board.is_game_over():
            self.over = True

    def on_back(self, event):
        if self.board.move_stack:
            move = self.board.pop()
            self.history.append(move)
            self.draw_board()

    def on_forward(self, event):
        if self.history:
            self.board.push(self.history.pop())
            self.draw_board()

    def update_textbox(self, text):
        self.textbox.config(state="normal")
        self.textbox.delete("1.0", tk.END)
        self.textbox.insert(tk.END, text)
        self.textbox.config(state="disabled")

    def get_action(self):
        return str(np.random.choice(list(self.board.legal_moves)))

class Play(ChessGUI):

    def __init__(self, agent):
        self.agent = agent
        super().__init__()
        
    def check_mate(self):
        board = self.board.copy()
        if board.move_stack:
            action = str(board.pop())        
            _, self.over = self.agent.model.game(board).get_value(action)
            if self.over:
                self.update_textbox("Game Over!")

    def get_action(self):
        return self.agent.model.game.int_to_action(np.argmax(self.agent.get_action(self.agent.model.game(self.board))))