"""Chess.com TCN: two characters per move, the compact move list in game JSON.

Decoding TCN is much faster than parsing PGN, so the dataset builder uses it
for moves and only scans the PGN for clock comments.
"""
import chess

ALPHABET = ('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!?'
            '{~}(^)[_]@#$,./&-*++=')
PROMOTIONS = 'qnrbkp'
_INDEX = {c: i for i, c in enumerate(ALPHABET[:64])}
_INDEX.update({c: 64 + i for i, c in enumerate(ALPHABET[64:]) if c not in _INDEX})
_PIECE = {'q': chess.QUEEN, 'n': chess.KNIGHT, 'r': chess.ROOK, 'b': chess.BISHOP}


def decode(tcn):
    """Return chess.Move objects; legality is checked when they are pushed."""
    moves = []
    for i in range(0, len(tcn), 2):
        frm, to = _INDEX[tcn[i]], _INDEX[tcn[i + 1]]
        if frm > 63:
            raise ValueError(f'drops are not standard chess: {tcn[i:i + 2]!r}')
        promotion = None
        if to > 63:
            promotion = _PIECE[PROMOTIONS[(to - 64) // 3]]
            to = frm + (-8 if frm < 16 else 8) + (to - 64) % 3 - 1
        moves.append(chess.Move(frm, to, promotion))
    return moves


def legalize(board, move):
    """Some older TCNs write a queen promotion as a plain from->to move."""
    if move.promotion is None and board.piece_type_at(move.from_square) == chess.PAWN \
            and chess.square_rank(move.to_square) in (0, 7):
        return chess.Move(move.from_square, move.to_square, chess.QUEEN)
    return move


def encode(moves):
    out = []
    for move in moves:
        to = move.to_square
        if move.promotion:
            step = to - move.from_square - (-8 if move.from_square < 16 else 8)
            to = 64 + 3 * PROMOTIONS.index(chess.piece_symbol(move.promotion)) + step + 1
        out.append(ALPHABET[move.from_square] + ALPHABET[to])
    return ''.join(out)
