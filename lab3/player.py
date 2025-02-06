from abc import ABC, abstractmethod
from game import TicTacToe

import numpy as np


def build_player(player_config, game):
    assert player_config["type"] in ["human", "random", "minimax"]

    if player_config["type"] == "human":
        return HumanPlayer(game)

    if player_config["type"] == "random":
        return RandomComputerPlayer(game)

    if player_config["type"] == "minimax":
        return MinimaxComputerPlayer(game, player_config)


class Player(ABC):
    def __init__(self, game: TicTacToe):
        self.game = game
        self.score = 0

    @abstractmethod
    def get_move(self, event_position):
        pass


class HumanPlayer(Player):
    def get_move(self, event_position):
        return event_position


class RandomComputerPlayer(Player):
    def get_move(self, event_position):
        available_moves = self.game.available_moves()
        move_id = np.random.choice(len(available_moves))
        return available_moves[move_id]


class MinimaxComputerPlayer(Player):
    def __init__(self, game, config):
        super().__init__(game)
        self.prune_depth = config["pruning_depth"]
        self.player = ""
        self.HEURISTIC = [[3, 2, 3], [2, 4, 2], [3, 2, 3]]

    def eval(self, is_maximizing: bool) -> int:
        winner = self.game.get_winner()
        if winner == self.player:
            return 10 if is_maximizing else -10
        elif winner in ["t", ""] or winner is None:
            return 0
        else:
            return -10 if is_maximizing else 10

    def minimax(
        self, depth: int, is_maximizing: bool, alpha: float, beta: float
    ) -> float:
        if depth == 0 or self.game.get_winner() != "":
            return self.eval(is_maximizing)

        if is_maximizing:
            max_eval = float("-inf")
            for move in self.game.available_moves():
                self.game.move(move)
                eval = self.minimax(depth - 1, not is_maximizing, alpha, beta)
                self.game.unmove()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for move in self.game.available_moves():
                self.game.move(move)
                eval = self.minimax(depth - 1, not is_maximizing, alpha, beta)
                self.game.unmove()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_move(self, event_position):
        available_moves = self.game.available_moves()
        self.player = self.game.get_player()

        best_move = None
        best_eval = float("inf")
        alpha = float("-inf")

        # return available_moves[0]
        for move in available_moves:
            self.game.move(move)
            move_value = self.minimax(self.prune_depth, True, alpha, -1 * alpha)
            self.game.unmove()

            if move_value < best_eval:
                best_move = move
                best_eval = move_value

        return best_move
