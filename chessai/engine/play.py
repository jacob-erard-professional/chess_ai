"""Play games between any two players and record them in chess.com format.

    python -m chessai.engine.play --white human --black alphabeta --tc 180+2
    python -m chessai.engine.play --white stockfish:elo=1500 --black random --games 10

Games are appended to data/played/YYYY-MM.json as {"games": [...]}, the same
layout as the monthly archives in data/players/, so the dataset builder reads
them with the same code.
"""
import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import chess

from .game import Game, PlayerInfo
from .players import RESIGN, from_spec

DEFAULT_OUT = Path('data/played')


def play_game(white, black, time_control='180+0', fen=chess.STARTING_FEN,
              ratings=(None, None), clock='wall', rated=False, max_plies=1000):
    """Run one game to the end and return the Game.

    clock='wall' charges each player the real time it spent choosing, so a
    slow model can lose on time. clock='free' charges nothing.
    """
    players = {chess.WHITE: white, chess.BLACK: black}
    game = Game(PlayerInfo(white.name, ratings[0]), PlayerInfo(black.name, ratings[1]),
                time_control=time_control, fen=fen, rated=rated)
    while not game.is_over and len(game.board.move_stack) < max_plies:
        color = game.turn
        started = time.monotonic()
        move = players[color].choose_move(game)
        elapsed = time.monotonic() - started if clock == 'wall' else 0.0
        if game.is_over:  # e.g. a human claimed a draw
            break
        if move == RESIGN:
            game.resign(color)
        else:
            game.play(move, elapsed)
    if not game.is_over:
        game.abandon(game.turn)
    return game


def save_game(record, out=DEFAULT_OUT):
    """Append to out/YYYY-MM.json atomically; returns the file path."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    month = dt.datetime.fromtimestamp(record['end_time'] or time.time(), dt.UTC).strftime('%Y-%m')
    path = out / f'{month}.json'
    data = json.loads(path.read_text()) if path.exists() else {'games': []}
    data['games'].append(record)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--white', default='human')
    ap.add_argument('--black', default='alphabeta')
    ap.add_argument('--tc', default='180+0', help="e.g. 60+1, 180+2, or '-' for untimed")
    ap.add_argument('--fen', default=chess.STARTING_FEN)
    ap.add_argument('--games', type=int, default=1)
    ap.add_argument('--swap', action='store_true', help='alternate colors each game')
    ap.add_argument('--clock', choices=('wall', 'free'), default='wall')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    a, b = from_spec(args.white), from_spec(args.black)
    score = {a.name: 0.0, b.name: 0.0}
    try:
        for i in range(args.games):
            white, black = (b, a) if args.swap and i % 2 else (a, b)
            game = play_game(white, black, args.tc, args.fen, clock=args.clock)
            if game.winner is None:
                score[white.name] += 0.5
                score[black.name] += 0.5
            else:
                score[(white if game.winner else black).name] += 1
            where = '' if args.no_save else f' -> {save_game(game.to_chesscom(), args.out)}'
            print(f'game {i + 1}: {white.name} vs {black.name} {game.result} '
                  f'({game.termination()}, {len(game.moves)} plies){where}')
    finally:
        a.close()
        b.close()
    if args.games > 1:
        print('score:', ', '.join(f'{k} {v}' for k, v in score.items()))


if __name__ == '__main__':
    main()
