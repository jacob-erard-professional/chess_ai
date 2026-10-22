"""Anything that picks moves. A model only needs `choose_move(game) -> Move`.

`choose_move` may also return RESIGN. Players see the whole Game (board,
clocks, history, player ratings), so a model can condition on all of it.
"""
import random
import shutil

import chess
import chess.engine

from .search import Searcher

RESIGN = 'resign'


class Player:
    name = 'player'

    def choose_move(self, game):
        raise NotImplementedError

    def close(self):
        pass


class RandomPlayer(Player):
    name = 'random'

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_move(self, game):
        return self.rng.choice(game.legal_moves())


class HumanPlayer(Player):
    """Reads SAN or UCI from the terminal. Also: 'resign', 'draw' (claim), 'moves'."""
    name = 'human'

    def choose_move(self, game):
        print(game.board.unicode(invert_color=True, borders=True,
                                 orientation=game.turn))
        if game.time_control:
            print(f"clocks  white {game.clocks[chess.WHITE]:.1f}s  black {game.clocks[chess.BLACK]:.1f}s")
        while True:
            text = input(f"{'White' if game.turn else 'Black'} to move: ").strip()
            if text == 'resign':
                return RESIGN
            if text == 'draw':
                if game.claim_draw():
                    return None
                print('no draw to claim')
                continue
            if text == 'moves':
                print(' '.join(game.board.san(m) for m in game.legal_moves()))
                continue
            try:
                move = game.parse_move(text)
                if move in game.board.legal_moves:
                    return move
            except ValueError:
                pass
            print(f'illegal or unreadable: {text!r}')


class AlphaBetaPlayer(Player):
    """The classical searcher in search.py.

    time=SECONDS per move, or ~1/30 of the clock in timed games. With a time
    budget and no explicit depth, it deepens until the time runs out;
    otherwise it searches to `depth` (default 4).
    """
    DEFAULT_DEPTH = 4
    TIMED_MAX_DEPTH = 64

    def __init__(self, depth=None, time=None):
        self.depth = int(depth) if depth else None
        self.time = float(time) if time else None
        self.name = f'alphabeta-d{self.depth}' if self.depth else \
            f'alphabeta-t{self.time:g}' if self.time else 'alphabeta'

    def choose_move(self, game):
        budget = self.time
        if game.time_control and budget is None:
            budget = game.clocks[game.turn] / 30 + game.time_control.increment * 0.8
        depth = self.depth or (self.TIMED_MAX_DEPTH if budget else self.DEFAULT_DEPTH)
        move, _ = Searcher(depth, budget).search(game.board)
        return move


class StockfishPlayer(Player):
    """Stockfish at full strength, or limited with elo= within the engine's
    UCI_Elo range (1320-3190 for Stockfish 16-19)."""

    def __init__(self, elo=None, depth=None, time=0.1, path=None):
        path = path or shutil.which('stockfish')
        if not path:
            raise RuntimeError('stockfish not found on PATH (brew install stockfish)')
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        if elo:
            option = self.engine.options['UCI_Elo']
            if not option.min <= int(elo) <= option.max:
                self.engine.quit()
                raise ValueError(f'stockfish elo must be {option.min}-{option.max}, got {elo}')
            self.engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': int(elo)})
        self.limit = chess.engine.Limit(depth=int(depth)) if depth else chess.engine.Limit(time=float(time))
        self.name = f'stockfish-{elo}' if elo else 'stockfish'

    def choose_move(self, game):
        return self.engine.play(game.board, self.limit).move

    def close(self):
        self.engine.quit()


def _model_player(**kwargs):
    from ..models.player import ModelPlayer  # imports torch, so only when asked for
    return ModelPlayer(**kwargs)


REGISTRY = {'random': RandomPlayer, 'human': HumanPlayer, 'alphabeta': AlphaBetaPlayer,
            'stockfish': StockfishPlayer, 'model': _model_player}


def from_spec(spec):
    """'alphabeta:depth=3,time=1' -> AlphaBetaPlayer(depth='3', time='1')."""
    kind, _, args = spec.partition(':')
    kwargs = dict(a.split('=', 1) for a in args.split(',') if a)
    if kind not in REGISTRY:
        raise ValueError(f'unknown player {kind!r}; choose from {sorted(REGISTRY)}')
    return REGISTRY[kind](**kwargs)
