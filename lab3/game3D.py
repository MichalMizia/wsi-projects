import numpy as np
from game import TicTacToe

N_ROWS = 4


class TicTacToe3D(TicTacToe):
    def __init__(self):
        self.winningCombinations = [
            # layer 1
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            [4, 8, 12, 16],
            [1, 6, 11, 16],
            [4, 7, 10, 13],
            # layer 2
            [17, 18, 19, 20],
            [21, 22, 23, 24],
            [25, 26, 27, 28],
            [29, 30, 31, 32],
            [17, 21, 25, 29],
            [18, 22, 26, 30],
            [19, 23, 27, 31],
            [20, 24, 28, 32],
            [17, 22, 27, 32],
            [20, 23, 26, 29],
            # layer 3
            [33, 34, 35, 36],
            [37, 38, 39, 40],
            [41, 42, 43, 44],
            [45, 46, 47, 48],
            [33, 37, 41, 45],
            [34, 38, 42, 46],
            [35, 39, 43, 47],
            [36, 40, 44, 48],
            [33, 38, 43, 48],
            [36, 39, 42, 45],
            # layer 4
            [49, 50, 51, 52],
            [53, 54, 55, 56],
            [57, 58, 59, 60],
            [61, 62, 63, 64],
            [49, 53, 57, 61],
            [50, 54, 58, 62],
            [51, 55, 59, 63],
            [52, 56, 60, 64],
            [49, 54, 59, 64],
            [52, 55, 58, 61],
            #  through all layers vertically
            [1, 17, 33, 49],
            [2, 18, 34, 50],
            [3, 19, 35, 51],
            [4, 20, 36, 52],
            [5, 21, 37, 53],
            [6, 22, 38, 54],
            [7, 23, 39, 55],
            [8, 24, 40, 56],
            [9, 25, 41, 57],
            [10, 26, 42, 58],
            [11, 27, 43, 59],
            [12, 28, 44, 60],
            [13, 29, 45, 61],
            [14, 30, 46, 62],
            [15, 31, 47, 63],
            [16, 32, 48, 64],
            #  through all layers vertically and forward/backward
            [1, 21, 41, 61],
            [2, 22, 42, 62],
            [3, 23, 43, 63],
            [4, 24, 44, 64],
            [13, 25, 37, 49],
            [14, 26, 38, 50],
            [15, 27, 39, 51],
            [16, 28, 40, 52],
            # through all layers vertically and side to side
            [1, 21, 41, 61],
            [2, 22, 42, 62],
            [3, 23, 43, 63],
            [4, 24, 44, 64],
            [13, 25, 37, 49],
            [14, 26, 38, 50],
            [15, 27, 39, 51],
            [16, 28, 40, 52],
            # long diagonals
            [1, 22, 43, 64],
            [4, 23, 42, 61],
        ]
        self.board = np.zeros((N_ROWS, N_ROWS, N_ROWS), dtype=np.str_)
        self.player_x_turn = True
        self.player_x_started = True
        self.last_moves = []
        self.winner = None  # winner is in ["x", "o", "t", None] t -> tie, None -> not finished yet

    def get_winner(self):
        for combination in self.winningCombinations:
            symbols = [
                self.board[
                    (pos - 1) // N_ROWS // N_ROWS,
                    (pos - 1) // N_ROWS % N_ROWS,
                    (pos - 1) % N_ROWS,
                ]
                for pos in combination
            ]
            if len(set(symbols)) == 1 and symbols[0] != "":
                return symbols[0]

        if np.all(self.board != ""):
            return "t"

        return ""
