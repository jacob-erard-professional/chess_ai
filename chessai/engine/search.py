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


def evaluate(board):
    """Centipawns for the side to move."""
    score = 0
    for square, piece in board.piece_map().items():
        index = square ^ 56 if piece.color == chess.WHITE else square
        value = VALUES[piece.piece_type] + _PST[piece.piece_type][index]
        score += value if piece.color == chess.WHITE else -value
    return score if board.turn == chess.WHITE else -score


class _Timeout(Exception):
    pass


class Searcher:
    def __init__(self, max_depth=4, time_limit=None):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.nodes = 0

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

    def quiesce(self, board, alpha, beta):
        self._tick()
        stand = evaluate(board)
        if stand >= beta:
            return beta
        alpha = max(alpha, stand)
        for move in self._ordered(board, [m for m in board.legal_moves if board.is_capture(m)]):
            board.push(move)
            score = -self.quiesce(board, -beta, -alpha)
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
            return self.quiesce(board, alpha, beta)
        for move in self._ordered(board, board.legal_moves):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def search(self, board):
        """Best (move, score) found within max_depth and time_limit."""
        board = board.copy()
        self.nodes = 0
        self.deadline = time.monotonic() + self.time_limit if self.time_limit else None
        moves = list(board.legal_moves)
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
            if abs(alpha) >= MATE - 100:
                break
        return best, best_score
