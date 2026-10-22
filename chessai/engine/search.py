"""Classical search: negamax alpha-beta with iterative deepening and quiescence.

Deliberately small and readable rather than strong. The evaluation is
material plus piece-square tables (Tomasz Michniewski's "simplified
evaluation function"), always from the side to move's point of view.
"""
import time

import chess

MATE = 100_000
VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# Tables are written rank 8 first (as seen from White), so index with square ^ 56.
_PST = {
    chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0,
                 50, 50, 50, 50, 50, 50, 50, 50,
                 10, 10, 20, 30, 30, 20, 10, 10,
                 5, 5, 10, 25, 25, 10, 5, 5,
                 0, 0, 0, 20, 20, 0, 0, 0,
                 5, -5, -10, 0, 0, -10, -5, 5,
                 5, 10, 10, -20, -20, 10, 10, 5,
                 0, 0, 0, 0, 0, 0, 0, 0],
    chess.KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50,
                   -40, -20, 0, 0, 0, 0, -20, -40,
                   -30, 0, 10, 15, 15, 10, 0, -30,
                   -30, 5, 15, 20, 20, 15, 5, -30,
                   -30, 0, 15, 20, 20, 15, 0, -30,
                   -30, 5, 10, 15, 15, 10, 5, -30,
                   -40, -20, 0, 5, 5, 0, -20, -40,
                   -50, -40, -30, -30, -30, -30, -40, -50],
    chess.BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20,
                   -10, 0, 0, 0, 0, 0, 0, -10,
                   -10, 0, 5, 10, 10, 5, 0, -10,
                   -10, 5, 5, 10, 10, 5, 5, -10,
                   -10, 0, 10, 10, 10, 10, 0, -10,
                   -10, 10, 10, 10, 10, 10, 10, -10,
                   -10, 5, 0, 0, 0, 0, 5, -10,
                   -20, -10, -10, -10, -10, -10, -10, -20],
    chess.ROOK: [0, 0, 0, 0, 0, 0, 0, 0,
                 5, 10, 10, 10, 10, 10, 10, 5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 0, 0, 0, 5, 5, 0, 0, 0],
    chess.QUEEN: [-20, -10, -10, -5, -5, -10, -10, -20,
                  -10, 0, 0, 0, 0, 0, 0, -10,
                  -10, 0, 5, 5, 5, 5, 0, -10,
                  -5, 0, 5, 5, 5, 5, 0, -5,
                  0, 0, 5, 5, 5, 5, 0, -5,
                  -10, 5, 5, 5, 5, 5, 0, -10,
                  -10, 0, 5, 0, 0, 0, 0, -10,
                  -20, -10, -10, -5, -5, -10, -10, -20],
    chess.KING: [-30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -20, -30, -30, -40, -40, -30, -30, -20,
                 -10, -20, -20, -20, -20, -20, -20, -10,
                 20, 20, 0, 0, 0, 0, 20, 20,
                 20, 30, 10, 0, 0, 10, 30, 20],
}


_KING_ENDGAME = [-50, -40, -30, -20, -20, -30, -40, -50,
                 -30, -20, -10, 0, 0, -10, -20, -30,
                 -30, -10, 20, 30, 30, 20, -10, -30,
                 -30, -10, 30, 40, 40, 30, -10, -30,
                 -30, -10, 30, 40, 40, 30, -10, -30,
                 -30, -10, 20, 30, 30, 20, -10, -30,
                 -30, -30, 0, 0, 0, 0, -30, -30,
                 -50, -30, -30, -30, -30, -30, -30, -50]

# Game phase: 24 with all pieces on the board, 0 with only kings and pawns.
_PHASE = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
FULL_PHASE = 24


def _center_distance(square):
    """Manhattan distance from the 2x2 centre: 0 on d4/e4/d5/e5, 6 in a corner."""
    f, r = chess.square_file(square), chess.square_rank(square)
    return max(3 - f, f - 4) + max(3 - r, r - 4)


def _mop_up(strong_king, weak_king):
    """Drive the lone king to the edge and bring our king closer."""
    return 10 * _center_distance(weak_king) + 4 * (14 - chess.square_manhattan_distance(strong_king, weak_king))


def evaluate(board):
    """Centipawns for the side to move.

    Material plus piece-square tables. The king table blends from middlegame
    to endgame as pieces come off. When one side has no pawns and is down at
    least a rook in pieces, the stronger side also gets a mop-up bonus,
    without which the search cannot find mates beyond its horizon.
    """
    score = phase = 0
    pieces = {chess.WHITE: 0, chess.BLACK: 0}
    pawns = {chess.WHITE: 0, chess.BLACK: 0}
    kings = {}
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        index = square ^ 56 if piece.color == chess.WHITE else square
        kind = piece.piece_type
        if kind == chess.KING:
            kings[piece.color] = square
            continue
        score += sign * (VALUES[kind] + _PST[kind][index])
        phase += _PHASE.get(kind, 0)
        if kind == chess.PAWN:
            pawns[piece.color] += 1
        else:
            pieces[piece.color] += VALUES[kind]
    phase = min(phase, FULL_PHASE)
    for color, square in kings.items():
        index = square ^ 56 if color == chess.WHITE else square
        king = (_PST[chess.KING][index] * phase + _KING_ENDGAME[index] * (FULL_PHASE - phase)) // FULL_PHASE
        score += king if color == chess.WHITE else -king
    for strong in (chess.WHITE, chess.BLACK):
        weak = not strong
        if len(kings) == 2 and not pawns[weak] and pieces[strong] - pieces[weak] >= VALUES[chess.ROOK]:
            bonus = _mop_up(kings[strong], kings[weak])
            score += bonus if strong == chess.WHITE else -bonus
    return score if board.turn == chess.WHITE else -score


class _Timeout(Exception):
    pass


class Searcher:
    def __init__(self, max_depth=4, time_limit=None):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.nodes = 0
        self.deadline = None
        self.depth_reached = 0

    def _ordered(self, board, moves, first=None):
        def key(m):
            if m == first:
                return -10_000
            victim = board.piece_type_at(m.to_square) or (chess.PAWN if board.is_en_passant(m) else None)
            if victim:  # MVV-LVA: big victim, small attacker first
                return -(10 * VALUES[victim] - VALUES[board.piece_type_at(m.from_square)]) - 1000
            return -VALUES[m.promotion] if m.promotion else 0
        return sorted(moves, key=key)

    def _tick(self):
        self.nodes += 1
        if self.deadline and self.nodes % 1024 == 0 and time.monotonic() > self.deadline:
            raise _Timeout

    def quiesce(self, board, alpha, beta, ply):
        """Captures only, until quiet. In check, every evasion is searched and
        there is no stand-pat, so mates at the horizon are seen."""
        self._tick()
        in_check = board.is_check()
        if not any(board.generate_legal_moves()):
            return -MATE + ply if in_check else 0
        if in_check:
            moves = list(board.legal_moves)
        else:
            stand = evaluate(board)
            if stand >= beta:
                return beta
            alpha = max(alpha, stand)
            moves = [m for m in board.legal_moves if board.is_capture(m)]
        for move in self._ordered(board, moves):
            board.push(move)
            score = -self.quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def negamax(self, board, depth, alpha, beta, ply):
        self._tick()
        if board.is_checkmate():
            return -MATE + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(2) \
                or board.halfmove_clock >= 100:
            return 0
        if depth == 0:
            return self.quiesce(board, alpha, beta, ply)
        for move in self._ordered(board, board.legal_moves):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def search(self, board):
        """Best (move, score) found within max_depth and time_limit.

        With no legal moves the result is (None, -MATE) when checkmated and
        (None, 0) when stalemated.
        """
        board = board.copy()
        self.nodes = 0
        self.deadline = time.monotonic() + self.time_limit if self.time_limit else None
        moves = list(board.legal_moves)
        if not moves:
            return None, -MATE if board.is_check() else 0
        best, best_score = moves[0], None
        for depth in range(1, self.max_depth + 1):
            try:
                alpha, iter_best = -MATE - 1, None
                for move in self._ordered(board, moves, best):
                    board.push(move)
                    score = -self.negamax(board, depth - 1, -MATE - 1, -alpha, 1)
                    board.pop()
                    if iter_best is None or score > alpha:
                        alpha, iter_best = score, move
            except _Timeout:
                break
            best, best_score = iter_best, alpha
            self.depth_reached = depth
            if abs(alpha) >= MATE - 100:
                break
        return best, best_score
