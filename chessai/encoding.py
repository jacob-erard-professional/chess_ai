"""The one place positions and moves become numbers. Every model uses these.

Everything is from the side to move's perspective: when Black is to move the
board is mirrored vertically and colors swap, so "own pieces move up the
board" always holds and a model never needs to learn chess twice.

Board: 64 uint8 piece codes, square a1=0 .. h8=63 in that perspective.
    0 empty, 1-6 own P N B R Q K, 7-12 opponent P N B R Q K.
Castling: 4 bits (own K-side, own Q-side, opp K-side, opp Q-side).
En passant: file 0-7, or NO_EP.
Move: index into MOVES (1858 entries): every from->to a queen or knight could
    travel on an empty board, plus knight/bishop/rook underpromotions. Queen
    promotion uses the plain from->to entry.
"""
import chess
import numpy as np

ENCODING_VERSION = 1
NO_EP = 8
PIECE_PLANES = 12
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _build_moves():
    moves = []
    for frm in range(64):
        f, r = frm % 8, frm // 8
        for to in range(64):
            if to == frm:
                continue
            df, dr = to % 8 - f, to // 8 - r
            if df == 0 or dr == 0 or abs(df) == abs(dr) or {abs(df), abs(dr)} == {1, 2}:
                moves.append(chess.Move(frm, to))
    for frm in range(48, 56):
        for df in (-1, 0, 1):
            if 0 <= frm % 8 + df < 8:
                for piece in UNDERPROMOTIONS:
                    moves.append(chess.Move(frm, frm + 8 + df, piece))
    return tuple(moves)


MOVES = _build_moves()
NUM_MOVES = len(MOVES)
MOVE_INDEX = {m: i for i, m in enumerate(MOVES)}


def _mirror(square):
    return square ^ 56


def move_to_index(move, turn):
    """Index of an absolute-coordinates move for the side `turn`."""
    frm, to = move.from_square, move.to_square
    if turn == chess.BLACK:
        frm, to = _mirror(frm), _mirror(to)
    promo = move.promotion if move.promotion != chess.QUEEN else None
    return MOVE_INDEX[chess.Move(frm, to, promo)]


def index_to_move(index, board):
    """Absolute-coordinates move on `board` for a vocabulary index."""
    m = MOVES[index]
    frm, to = m.from_square, m.to_square
    if board.turn == chess.BLACK:
        frm, to = _mirror(frm), _mirror(to)
    promo = m.promotion
    if promo is None and board.piece_type_at(frm) == chess.PAWN and chess.square_rank(to) in (0, 7):
        promo = chess.QUEEN
    return chess.Move(frm, to, promo)


def encode_board(board):
    """(pieces uint8[64], castling uint8 bitmask, ep file uint8)."""
    pieces = np.zeros(64, np.uint8)
    us = board.turn
    flip = us == chess.BLACK
    for square, piece in board.piece_map().items():
        code = piece.piece_type + (0 if piece.color == us else 6)
        pieces[_mirror(square) if flip else square] = code
    castling = 0
    them = not us
    for bit, (color, side) in enumerate(((us, 'k'), (us, 'q'), (them, 'k'), (them, 'q'))):
        has = board.has_kingside_castling_rights if side == 'k' else board.has_queenside_castling_rights
        castling |= has(color) << bit
    ep = chess.square_file(board.ep_square) if board.has_legal_en_passant() else NO_EP
    return pieces, castling, ep


def legal_mask(board):
    """bool[NUM_MOVES]: which vocabulary entries are legal here."""
    mask = np.zeros(NUM_MOVES, bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board.turn)] = True
    return mask


def board_planes(pieces, castling, ep):
    """float32 planes (N, 17, 8, 8) from batched or single encoded boards.

    Planes 0-11: one per piece code. 12-15: castling rights (constant planes).
    16: en passant target file, marked on the rank the capturing pawn lands on.
    Plane row = rank, column = file.
    """
    pieces = np.asarray(pieces).reshape(-1, 64)
    n = len(pieces)
    planes = np.zeros((n, 17, 64), np.float32)
    rows, cols = np.nonzero(pieces)
    planes[rows, pieces[rows, cols] - 1, cols] = 1
    castling = np.asarray(castling).reshape(n)
    for bit in range(4):
        planes[:, 12 + bit] = ((castling >> bit) & 1)[:, None]
    ep = np.asarray(ep).reshape(n)
    has = ep != NO_EP
    planes[np.nonzero(has)[0], 16, 40 + ep[has]] = 1
    return planes.reshape(n, 17, 8, 8)


def decode_board(pieces, castling, ep):
    """Inverse of encode_board, as a chess.Board with White to move (the mover).

    For inspection and debugging: the result is in the mover's perspective.
    """
    board = chess.Board(None)
    for square, code in enumerate(pieces):
        if code:
            code = int(code)
            board.set_piece_at(square, chess.Piece((code - 1) % 6 + 1, code <= 6))
    rights = ''.join(c for bit, c in enumerate('KQkq') if int(castling) >> bit & 1)
    board.set_castling_fen(rights or '-')
    board.ep_square = 40 + int(ep) if ep != NO_EP else None
    return board
