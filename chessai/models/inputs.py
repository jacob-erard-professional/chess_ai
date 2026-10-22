"""Model inputs for a live Game, built the same way the dataset builds them."""
import chess
import numpy as np

from .. import encoding
from ..dataset.reader import features, stack_planes
from ..dataset.schema import ROW, TIME_CLASSES, UNKNOWN_CLASS, pack_pieces


def _set_board(row, board):
    pieces, row['castling'], row['ep'] = encoding.encode_board(board)
    row['pieces'] = pack_pieces(pieces)[0]


def game_inputs(game, history):
    """(planes (1, C, 8, 8), features (1, F)) for the side to move in `game`."""
    board = game.board.copy()
    us = board.turn
    rows = np.zeros(history + 1, ROW)
    _set_board(rows[0:1][0], board)
    cur = rows[0:1]
    cur['halfmove'] = min(board.halfmove_clock, 255)
    cur['ply'] = len(board.move_stack)

    key, seen, replay = board._transposition_key(), 1, board.copy()
    while replay.move_stack:
        replay.pop()
        seen += replay._transposition_key() == key
    cur['repetition'] = min(seen, 255)

    me, them = game.players[us], game.players[not us]
    cur['elo'], cur['opp_elo'] = me.rating or 0, them.rating or 0
    tc = game.time_control
    cur['clock'] = game.clocks[us] if tc else np.nan
    cur['opp_clock'] = game.clocks[not us] if tc else np.nan
    cur['base'] = min(int(tc.base), 65535) if tc else 0
    cur['inc'] = min(int(tc.increment), 255) if tc else 0
    cur['time_class'] = TIME_CLASSES.index(tc.time_class) if tc else UNKNOWN_CLASS

    earlier = []
    for back in range(1, history + 1):
        valid = bool(board.move_stack)
        if valid:
            board.pop()
            _set_board(rows[back:back + 1][0], board)
        earlier.append((rows[back:back + 1], np.array([valid])))
    return stack_planes(cur, earlier), features(cur)


def legal_mask(game):
    return encoding.legal_mask(game.board)[None]
