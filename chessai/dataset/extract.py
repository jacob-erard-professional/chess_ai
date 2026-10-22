"""Game JSON (chess.com format) -> training rows."""
import collections
import hashlib

import chess
import numpy as np

from .. import encoding
from ..engine import tcn
from ..engine.game import DRAW_CODES, TimeControl, parse_clocks
from .schema import (NO_EVAL, NO_MOVE, ROW, TEST, TIME_CLASSES, TRAIN, UNKNOWN_CLASS, VAL,
                     pack_pieces)

RESULT_POINTS = {'1-0': {chess.WHITE: 1, chess.BLACK: -1},
                 '0-1': {chess.WHITE: -1, chess.BLACK: 1}}


class Skip(Exception):
    """A game that cannot become rows (variant, daily, no moves, bad data)."""


def hash_split(game_id, val=5, test=5):
    bucket = int(hashlib.sha1(game_id.encode()).hexdigest()[:8], 16) % 100
    return TEST if bucket < test else VAL if bucket < test + val else TRAIN


def game_result(data):
    w, b = data['white'].get('result'), data['black'].get('result')
    if w == 'win':
        return '1-0'
    if b == 'win':
        return '0-1'
    if w == b and w in DRAW_CODES:
        return '1/2-1/2'
    raise Skip(f'unknown result: {w}/{b}')


def game_meta(data):
    w, b = data['white'], data['black']
    return {'uuid': data.get('uuid'), 'url': data.get('url'),
            'white': w.get('username'), 'black': b.get('username'),
            'white_elo': w.get('rating'), 'black_elo': b.get('rating'),
            'time_control': data.get('time_control'), 'time_class': data.get('time_class'),
            'rated': data.get('rated'), 'end_time': data.get('end_time'),
            'result': game_result(data),
            'white_result': w.get('result'), 'black_result': b.get('result')}


def extract(data, game_index, split, targets=frozenset(), legal=False):
    """Rows (and optionally packed legal-move masks) for one game.

    targets: lowercase usernames whose moves get is_target=1.
    """
    if data.get('rules', 'chess') != 'chess':
        raise Skip(f"variant {data.get('rules')}")
    try:
        tc = TimeControl.parse(data.get('time_control'))
    except ValueError as e:
        raise Skip(str(e))
    if not data.get('tcn'):
        raise Skip('no moves')
    try:
        moves = tcn.decode(data['tcn'])
    except (KeyError, ValueError) as e:
        raise Skip(f'bad tcn: {e}')
    n = len(moves)
    if n == 0:
        raise Skip('no moves')
    board = chess.Board(data.get('initial_setup') or chess.STARTING_FEN)
    clocks = parse_clocks(data.get('pgn', ''))
    timed = tc is not None and len(clocks) == n

    users = {chess.WHITE: data['white'], chess.BLACK: data['black']}
    elo = {c: users[c].get('rating') or 0 for c in users}
    target = {c: (users[c].get('username') or '').lower() in targets for c in users}
    points = RESULT_POINTS.get(game_result(data), {chess.WHITE: 0, chess.BLACK: 0})
    left = {chess.WHITE: tc.base if timed else np.nan, chess.BLACK: tc.base if timed else np.nan}
    time_class = data.get('time_class')
    time_class = TIME_CLASSES.index(time_class) if time_class in TIME_CLASSES else UNKNOWN_CLASS

    pieces = np.zeros((n, 64), np.uint8)
    cols = {k: np.zeros(n, ROW[k]) for k in ('castling', 'ep', 'halfmove', 'repetition', 'move',
                                             'elo', 'opp_elo', 'clock', 'opp_clock', 'move_time',
                                             'result', 'is_target')}
    masks = np.zeros((n, encoding.NUM_MOVES), bool) if legal else None
    seen = collections.Counter()
    for i, move in enumerate(moves):
        us = board.turn
        move = tcn.legalize(board, move)
        if not board.is_legal(move):
            raise Skip(f'illegal move at ply {i + 1}: {move.uci()}')
        key = board._transposition_key()
        seen[key] += 1
        pieces[i], cols['castling'][i], cols['ep'][i] = encoding.encode_board(board)
        cols['halfmove'][i] = min(board.halfmove_clock, 255)
        cols['repetition'][i] = min(seen[key], 255)
        cols['move'][i] = encoding.move_to_index(move, us)
        cols['elo'][i], cols['opp_elo'][i] = elo[us], elo[not us]
        cols['clock'][i], cols['opp_clock'][i] = left[us], left[not us]
        if timed:
            cols['move_time'][i] = max(0.0, left[us] - clocks[i] + tc.increment)
            left[us] = clocks[i]
        else:
            cols['move_time'][i] = np.nan
        cols['result'][i] = points[us]
        cols['is_target'][i] = target[us]
        if legal:
            masks[i] = encoding.legal_mask(board)
        board.push(move)

    rows = np.zeros(n, ROW)
    rows['pieces'] = pack_pieces(pieces)
    for k, v in cols.items():
        rows[k] = v
    rows['ply'] = np.arange(n)
    rows['base'] = min(int(tc.base), 65535) if tc else 0
    rows['inc'] = min(int(tc.increment), 255) if tc else 0
    rows['time_class'] = time_class
    rows['split'] = split
    rows['game'] = game_index
    rows['eval_cp'] = rows['played_cp'] = NO_EVAL
    rows['best_move'] = NO_MOVE
    return rows, (np.packbits(masks, axis=1) if legal else None)
