"""On-disk training row: one per position where a move was played.

Rows for a game are contiguous and in ply order, and a game never spans two
shards, so the previous positions of row i are rows i-1, i-2, ... while
`game` matches. All per-player fields are from the mover's point of view.
"""
import numpy as np

ROW = np.dtype([
    ('pieces', 'u1', 32),   # encoding.encode_board pieces, two 4-bit codes per byte
    ('castling', 'u1'), ('ep', 'u1'),
    ('halfmove', 'u1'),     # 50-move counter, capped at 255
    ('repetition', 'u1'),   # times this position has occurred, including now
    ('ply', 'u2'),          # 0-based index of the move in the game
    ('move', 'u2'),         # encoding.MOVES index of the move played
    ('elo', 'u2'), ('opp_elo', 'u2'),             # 0 = unknown
    ('clock', 'f4'), ('opp_clock', 'f4'),         # seconds left before the move; NaN unknown
    ('move_time', 'f4'),                           # seconds spent on this move; NaN unknown
    ('base', 'u2'), ('inc', 'u1'), ('time_class', 'u1'),
    ('result', 'i1'),       # final result for the mover: 1 win, 0 draw, -1 loss
    ('is_target', 'u1'),    # mover is one of the build's target players
    ('split', 'u1'),        # TRAIN / VAL / TEST, assigned per game
    ('game', 'u4'),         # line number in games.jsonl
    ('eval_cp', 'i2'),      # Stockfish eval before the move (mover's view), NO_EVAL if absent
    ('played_cp', 'i2'),    # Stockfish eval of the played move (mover's view)
    ('best_move', 'u2'),    # Stockfish best move index, NO_MOVE if absent
])

TRAIN, VAL, TEST = 0, 1, 2
SPLITS = {'train': TRAIN, 'val': VAL, 'test': TEST}
TIME_CLASSES = ('bullet', 'blitz', 'rapid')
UNKNOWN_CLASS = 255
NO_EVAL = -32768
NO_MOVE = 65535
EVAL_CLIP = 30000  # mate scores land near +-30000 (see dataset.evals)


def pack_pieces(pieces):
    pieces = np.asarray(pieces, np.uint8).reshape(-1, 32, 2)
    return pieces[..., 0] | (pieces[..., 1] << 4)


def unpack_pieces(packed):
    packed = np.asarray(packed, np.uint8)
    return np.stack([packed & 15, packed >> 4], -1).reshape(*packed.shape[:-1], 64)
