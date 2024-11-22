import argparse
import json
import pathlib

import numpy as np

from gui import GameGUI
from game import TicTacToe
from player import build_player


def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)


def simulate_game(game, player_x, player_o, x_starting=True):
    current = player_x if x_starting else player_o

    while game.get_winner() == "":
        move = current.get_move(0)  # type: ignore
        game.move(move)
        current = player_o if current == player_x else player_x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=pathlib.Path, required=True, help="Path to game config"
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    with open(args.config, "r") as f:
        config = json.load(f)

    game = TicTacToe()
    player_x = build_player(config["x"], game)
    player_o = build_player(config["o"], game)

    if config["gui"]:
        gui = GameGUI(game, player_x, player_o)
        gui.mainloop()
    else:
        player_x = build_player(config["minimax"], game)
        player_o = build_player(config["random"], game)
        if player_o is None or player_x is None:
            exit()

        wins = {
            "x_starting": {"o": 0, "x": 0, "t": 0},
            "o_starting": {"o": 0, "x": 0, "t": 0},
        }

        for i in range(100):
            simulate_game(game, player_x, player_o, i % 2 == 0)
            # print(game.get_winner())
            wins["x_starting" if i % 2 == 0 else "o_starting"][game.get_winner()] += 1
            game.play_again()

        print(wins)
