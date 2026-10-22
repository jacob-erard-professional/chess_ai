"""A chess game with clocks, every standard rule, and chess.com-format records.

python-chess handles move generation and legality (castling, en passant,
promotion, check). This module adds the game layer around it: clocks with
increment, flagging, resignation, draw agreement, automatic draws, and
reading/writing the same game JSON the collector stores in data/.
"""
import datetime as dt
import re
import time
import uuid
from dataclasses import dataclass

import chess

from . import tcn

DRAW_CODES = {'stalemate': 'stalemate', 'repetition': 'repetition',
              'insufficient': 'insufficient material', '50move': '50-move rule',
              'agreed': 'agreement', 'timevsinsufficient': 'timeout vs insufficient material'}
WIN_CODES = {'checkmated': 'by checkmate', 'resigned': 'by resignation',
             'timeout': 'on time', 'abandoned': '- game abandoned'}
_CLOCK = re.compile(r'\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]')
_HEADER_DATE = re.compile(r'\[UTCDate "([^"]+)"\]')
_HEADER_TIME = re.compile(r'\[UTCTime "([^"]+)"\]')


class IllegalMove(ValueError):
    pass


class GameOver(RuntimeError):
    pass


@dataclass(frozen=True)
class TimeControl:
    base: float
    increment: float = 0.0

    @classmethod
    def parse(cls, text):
        """'180+2' or '60'; '-' or None means untimed."""
        if text in (None, '', '-'):
            return None
        if isinstance(text, TimeControl):
            return text
        base, _, inc = str(text).partition('+')
        if '/' in base:
            raise ValueError(f'daily time controls are not supported: {text!r}')
        try:
            return cls(float(base), float(inc or 0))
        except ValueError:
            raise ValueError(f'unknown time control: {text!r}') from None

    @property
    def time_class(self):
        # Chess.com classifies by expected duration of a 40-move game.
        expected = self.base + 40 * self.increment
        return 'bullet' if expected < 180 else 'blitz' if expected < 600 else 'rapid'

    def __str__(self):
        base, inc = _num(self.base), _num(self.increment)
        return f'{base}+{inc}' if self.increment else base


@dataclass(frozen=True)
class PlayerInfo:
    username: str
    rating: int | None = None


def _num(x):
    return str(int(x)) if float(x).is_integer() else str(x)


def format_clock(seconds):
    tenths = max(0, round(seconds * 10))
    h, rem = divmod(tenths, 36000)
    m, rem = divmod(rem, 600)
    s, d = divmod(rem, 10)
    return f'{h}:{m:02d}:{s:02d}' + (f'.{d}' if d else '')


def parse_clocks(pgn):
    return [int(h) * 3600 + int(m) * 60 + float(s) for h, m, s in _CLOCK.findall(pgn)]


def chesscom_insufficient(board):
    """Chess.com's automatic insufficient-material draw (since 2019).

    Beyond python-chess's dead positions, chess.com also ends games where
    neither side has a pawn, rook or queen and each has at most one minor
    piece (KB v KN, KN v KN, KB v KB on either colors), and KNN v K. Mates
    there need the loser's help, so FIDE rules play on.
    """
    if board.pawns | board.rooks | board.queens:
        return False
    minors = {c: chess.popcount(board.occupied_co[c] & (board.knights | board.bishops)) for c in chess.COLORS}
    if max(minors.values()) <= 1:
        return True
    return any(minors[c] == 2 == chess.popcount(board.occupied_co[c] & board.knights) and minors[not c] == 0
               for c in chess.COLORS)


class Game:
    """One game. `play` is the only way moves enter, so every rule is enforced.

    `auto_draw=True` matches chess.com: threefold repetition and the 50-move
    rule end the game immediately. With `auto_draw=False` they must be claimed
    via `claim_draw`, and only fivefold repetition and the 75-move rule are
    automatic, as in FIDE rules.
    """

    def __init__(self, white=PlayerInfo('White'), black=PlayerInfo('Black'),
                 time_control='180+0', fen=chess.STARTING_FEN, rated=False,
                 auto_draw=True, start_time=None):
        self.players = {chess.WHITE: white, chess.BLACK: black}
        self.time_control = TimeControl.parse(time_control)
        self.initial_fen = fen
        self.board = chess.Board(fen)
        self.rated = rated
        self.auto_draw = auto_draw
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time = None
        self.uuid = str(uuid.uuid4())
        self.url = None
        base = self.time_control.base if self.time_control else None
        self.clocks = {chess.WHITE: base, chess.BLACK: base}
        self.clock_history = []  # mover's clock after each move, increment included
        self.result = None       # '1-0', '0-1' or '1/2-1/2'
        self.reason = None       # chess.com code: 'checkmated', 'repetition', ...
        self.winner = None       # chess.WHITE, chess.BLACK, or None for a draw

    # --- state -----------------------------------------------------------
    @property
    def turn(self):
        return self.board.turn

    @property
    def is_over(self):
        return self.result is not None

    @property
    def moves(self):
        return list(self.board.move_stack)

    def legal_moves(self):
        return [] if self.is_over else list(self.board.legal_moves)

    def move_times(self):
        """Seconds spent on each move, or None per move when untimed."""
        if not self.time_control or len(self.clock_history) != len(self.board.move_stack):
            return [None] * len(self.board.move_stack)
        last = {chess.WHITE: self.time_control.base, chess.BLACK: self.time_control.base}
        color = chess.Board(self.initial_fen).turn
        out = []
        for clock in self.clock_history:
            out.append(round(last[color] - clock + self.time_control.increment, 3))
            last[color] = clock
            color = not color
        return out

    # --- actions ---------------------------------------------------------
    def parse_move(self, move):
        if isinstance(move, chess.Move):
            return move
        try:
            return self.board.parse_uci(move)
        except ValueError:
            try:
                return self.board.parse_san(move)
            except ValueError as e:
                raise IllegalMove(f'{move!r} in {self.board.fen()}') from e

    def play(self, move, elapsed=0.0):
        """Apply a move after `elapsed` seconds of thinking. Returns the Move.

        If the mover's clock runs out first, the game ends on time and the move
        is not played (the return value is then None).
        """
        if self.is_over:
            raise GameOver(f'game already ended: {self.result} ({self.reason})')
        move = self.parse_move(move)
        if move not in self.board.legal_moves:
            raise IllegalMove(f'{move.uci()} in {self.board.fen()}')
        color = self.board.turn
        if self.time_control:
            remaining = self.clocks[color] - elapsed
            if remaining <= 0:
                self.flag(color)
                return None
            self.clocks[color] = remaining + self.time_control.increment
            self.clock_history.append(self.clocks[color])
        self.board.push(move)
        self._check_automatic_end()
        return move

    def flag(self, color):
        """`color` ran out of time.

        Chess.com draws when the opponent has only a lone king or king plus one
        minor piece, even if a helpmate exists (python-chess's
        has_insufficient_material would call those wins).
        """
        self.clocks[color] = 0.0
        b = self.board
        occ = b.occupied_co[not color]
        if not occ & (b.pawns | b.rooks | b.queens) and chess.popcount(occ & (b.knights | b.bishops)) <= 1:
            self._finish(None, 'timevsinsufficient')
        else:
            self._finish(not color, 'timeout')

    def resign(self, color):
        self._require_live()
        self._finish(not color, 'resigned')

    def abandon(self, color):
        self._require_live()
        self._finish(not color, 'abandoned')

    def agree_draw(self):
        self._require_live()
        self._finish(None, 'agreed')

    def claim_draw(self):
        """Claim threefold repetition or the 50-move rule. Returns success."""
        self._require_live()
        if self.board.is_repetition(3):
            self._finish(None, 'repetition')
        elif self.board.halfmove_clock >= 100:
            self._finish(None, '50move')
        else:
            return False
        return True

    def _require_live(self):
        if self.is_over:
            raise GameOver(f'game already ended: {self.result} ({self.reason})')

    def _check_automatic_end(self):
        b = self.board
        if b.is_checkmate():
            self._finish(not b.turn, 'checkmated')
        elif b.is_stalemate():
            self._finish(None, 'stalemate')
        elif b.is_insufficient_material() or (self.auto_draw and chesscom_insufficient(b)):
            self._finish(None, 'insufficient')
        elif b.is_fivefold_repetition() or (self.auto_draw and b.is_repetition(3)):
            self._finish(None, 'repetition')
        elif b.is_seventyfive_moves() or (self.auto_draw and b.halfmove_clock >= 100):
            self._finish(None, '50move')

    def _finish(self, winner, reason, end_time=None):
        self.winner = winner
        self.reason = reason
        self.result = '1/2-1/2' if winner is None else '1-0' if winner else '0-1'
        self.end_time = end_time if end_time is not None else time.time()

    # --- chess.com record format ------------------------------------------
    def player_result(self, color):
        if self.result is None:
            return None
        if self.winner is None:
            return self.reason
        return 'win' if self.winner == color else self.reason

    def termination(self):
        if self.result is None:
            return 'Game in progress'
        if self.winner is None:
            return f'Game drawn by {DRAW_CODES.get(self.reason, self.reason)}'
        return f'{self.players[self.winner].username} won {WIN_CODES.get(self.reason, self.reason)}'

    def pgn(self):
        start = dt.datetime.fromtimestamp(self.start_time, dt.UTC)
        end = dt.datetime.fromtimestamp(self.end_time or time.time(), dt.UTC)
        w, b = self.players[chess.WHITE], self.players[chess.BLACK]
        headers = [('Event', 'Local Game'), ('Site', 'chess_ai'),
                   ('Date', start.strftime('%Y.%m.%d')), ('Round', '-'),
                   ('White', w.username), ('Black', b.username),
                   ('Result', self.result or '*'), ('CurrentPosition', self.board.fen(en_passant='fen'))]
        if self.initial_fen != chess.STARTING_FEN:
            headers += [('SetUp', '1'), ('FEN', self.initial_fen)]
        headers += [('Timezone', 'UTC'), ('UTCDate', start.strftime('%Y.%m.%d')),
                    ('UTCTime', start.strftime('%H:%M:%S'))]
        if w.rating is not None:
            headers.append(('WhiteElo', str(w.rating)))
        if b.rating is not None:
            headers.append(('BlackElo', str(b.rating)))
        headers += [('TimeControl', str(self.time_control or '-')),
                    ('Termination', self.termination()),
                    ('StartTime', start.strftime('%H:%M:%S')),
                    ('EndDate', end.strftime('%Y.%m.%d')), ('EndTime', end.strftime('%H:%M:%S'))]
        if self.url:
            headers.append(('Link', self.url))
        lines = [f'[{k} "{v}"]' for k, v in headers]

        replay = chess.Board(self.initial_fen)
        clocks = self.clock_history if len(self.clock_history) == len(self.board.move_stack) else None
        tokens = []
        for i, move in enumerate(self.board.move_stack):
            number = replay.fullmove_number
            prefix = f'{number}.' if replay.turn else f'{number}...'
            tokens.append(f'{prefix} {replay.san(move)}')
            if clocks:
                tokens.append(f'{{[%clk {format_clock(clocks[i])}]}}')
            replay.push(move)
        tokens.append(self.result or '*')
        return '\n'.join(lines) + '\n\n' + ' '.join(tokens) + '\n'

    def to_chesscom(self):
        """The same JSON shape as the games in data/ (chess.com API format)."""
        tc = self.time_control
        side = lambda c: {'rating': self.players[c].rating, 'result': self.player_result(c),
                          'username': self.players[c].username}
        return {
            'url': self.url, 'pgn': self.pgn(), 'time_control': str(tc or '-'),
            'end_time': int(self.end_time) if self.end_time else None, 'rated': self.rated,
            'tcn': tcn.encode(self.board.move_stack), 'uuid': self.uuid,
            'initial_setup': self.initial_fen, 'fen': self.board.fen(en_passant='fen'),
            'time_class': tc.time_class if tc else 'untimed', 'rules': 'chess',
            'white': side(chess.WHITE), 'black': side(chess.BLACK),
        }

    @classmethod
    def from_chesscom(cls, data):
        """Rebuild a recorded game, clocks and outcome included."""
        if data.get('rules', 'chess') != 'chess':
            raise ValueError(f"unsupported variant: {data.get('rules')!r}")
        w, b = data['white'], data['black']
        game = cls(PlayerInfo(w['username'], w.get('rating')), PlayerInfo(b['username'], b.get('rating')),
                   time_control=data.get('time_control'),
                   fen=data.get('initial_setup') or chess.STARTING_FEN,
                   rated=data.get('rated', False), auto_draw=False,
                   start_time=_pgn_start_time(data.get('pgn', '')) or data.get('end_time'))
        game.uuid, game.url = data.get('uuid', game.uuid), data.get('url')
        moves = tcn.decode(data['tcn']) if data.get('tcn') else _pgn_moves(data['pgn'])
        clocks = parse_clocks(data.get('pgn', ''))
        if len(clocks) != len(moves):
            clocks = None
        for i, move in enumerate(moves):
            move = tcn.legalize(game.board, move)
            if move not in game.board.legal_moves:
                raise IllegalMove(f'ply {i + 1}: {move.uci()} in {game.board.fen()}')
            game.board.push(move)
            if clocks and game.time_control:
                game.clock_history.append(clocks[i])
                game.clocks[not game.board.turn] = clocks[i]
        game._set_recorded_outcome(w, b, data.get('end_time'))
        return game

    def _set_recorded_outcome(self, w, b, end_time):
        wr, br = w.get('result'), b.get('result')
        if wr == 'win':
            self._finish(chess.WHITE, br, end_time)
        elif br == 'win':
            self._finish(chess.BLACK, wr, end_time)
        elif wr:
            self._finish(None, wr, end_time)


def _pgn_start_time(pgn):
    date, clock = _HEADER_DATE.search(pgn), _HEADER_TIME.search(pgn)
    if not (date and clock):
        return None
    start = dt.datetime.strptime(f'{date[1]} {clock[1]}', '%Y.%m.%d %H:%M:%S')
    return start.replace(tzinfo=dt.UTC).timestamp()


def _pgn_moves(pgn):
    import io
    import chess.pgn
    return list(chess.pgn.read_game(io.StringIO(pgn)).mainline_moves())
