#!/usr/bin/env python3
"""Resumable local analysis of Chess.com monthly JSON archives."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

VERSION = 'legal-sacrifice-v1'
VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
_ENGINE = None
_SETTINGS = None
_ENGINE_PATH = None


def captured_value(board, move):
    if board.is_en_passant(move):
        return 100
    piece = board.piece_at(move.to_square)
    return VALUES[piece.piece_type] if piece else 0


def exchange_gain(board, square):
    """Best material gain from a legal exchange on square; either side may stop.

    Includes pinned-piece legality, king safety, promotions and en passant.
    Only captures on this square are modeled; tactical compensation is evaluated
    separately by Stockfish. Each recursion removes a piece, so it terminates.
    """
    best = 0
    for move in list(board.legal_moves):
        if move.to_square != square or not board.is_capture(move):
            continue
        gain = captured_value(board, move)
        if move.promotion:
            gain += VALUES[move.promotion] - 100
        board.push(move)
        gain -= exchange_gain(board, square)
        board.pop()
        best = max(best, gain)
    return best


def sacrifice_evidence(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type in (chess.PAWN, chess.KING) or move.promotion:
        return None
    gained = captured_value(board, move)
    after = board.copy()
    after.push(move)
    if after.is_checkmate():
        return None
    offered = exchange_gain(after, move.to_square)
    net = offered - gained
    if net < 100:
        return None
    return {'piece': piece.symbol(), 'square': chess.square_name(move.to_square),
            'exchange_gain_for_opponent_cp': offered,
            'material_captured_cp': gained, 'net_material_offered_cp': net,
            'scope': 'legal exchanges on moved piece destination'}


def score_data(score, turn):
    value = score.pov(turn)
    return {'cp': value.score(), 'mate': value.mate(),
            'ordering_cp': value.score(mate_score=100000),
            'perspective': 'side_to_move_in_fen'}


def line_data(board, info):
    pv = info.get('pv', [])
    if not pv:
        raise RuntimeError('Engine returned no principal variation')
    return {'uci': pv[0].uci(), 'san': board.san(pv[0]),
            'score': score_data(info['score'], board.turn),
            'depth': info.get('depth'), 'seldepth': info.get('seldepth'),
            'nodes': info.get('nodes'), 'pv_uci': [m.uci() for m in pv],
            'lowerbound': bool(info.get('lowerbound', False)),
            'upperbound': bool(info.get('upperbound', False))}


def close_engine():
    if _ENGINE is not None:
        try:
            _ENGINE.quit()
        except chess.engine.EngineTerminatedError:
            pass


def init_worker(engine_path, settings):
    global _ENGINE_PATH, _SETTINGS
    _ENGINE_PATH, _SETTINGS = engine_path, settings


def analyze_position(task):
    global _ENGINE
    # Explicit lifetime avoids idle engine threads preventing worker shutdown.
    with chess.engine.SimpleEngine.popen_uci(_ENGINE_PATH) as engine:
        _ENGINE = engine
        engine.configure({'Threads': 1, 'Hash': 128})
        return analyze_position_with_engine(task)


def analyze_position_with_engine(task):
    """Every legal root move is required before a position can be committed."""
    key, start_fen, history, played_uci = task
    board = chess.Board(start_fen)
    for uci in history:
        board.push_uci(uci)
    legal = list(board.legal_moves)
    if not legal:
        raise RuntimeError('No legal moves at played position')
    # Clear between positions for reproducibility; pass full move history for draws.
    _ENGINE.configure({'Clear Hash': None})
    infos = _ENGINE.analyse(board, chess.engine.Limit(depth=_SETTINGS['depth'], time=_SETTINGS['seconds']),
                            multipv=len(legal), game=key)
    lines = {info['pv'][0].uci(): line_data(board, info) for info in infos if info.get('pv')}
    # A time limit can end an incomplete MultiPV iteration. Explicitly search any
    # missing move rather than silently calling the worst reported line "worst".
    for move in legal:
        if move.uci() not in lines:
            info = _ENGINE.analyse(board, chess.engine.Limit(depth=_SETTINGS['depth'], time=_SETTINGS['seconds']),
                                   root_moves=[move], game=key)
            lines[move.uci()] = line_data(board, info)
    if set(lines) != {m.uci() for m in legal}:
        raise RuntimeError('Incomplete legal-move coverage')
    ranked = sorted(lines.values(), key=lambda x: (-x['score']['ordering_cp'], x['uci']))
    played = lines[played_uci]
    best_score = ranked[0]['score']['ordering_cp']
    played_score = played['score']['ordering_cp']
    evidence = sacrifice_evidence(board, chess.Move.from_uci(played_uci))
    brilliant = None
    if evidence and best_score - played_score <= 75 and -150 <= played_score:
        # Confirm with unrestricted deep analysis and a forced played-move search.
        limit = chess.engine.Limit(depth=_SETTINGS['confirm_depth'], time=_SETTINGS['confirm_seconds'])
        top_info = _ENGINE.analyse(board, limit, game=key)
        played_info = _ENGINE.analyse(board, limit, root_moves=[chess.Move.from_uci(played_uci)], game=key)
        top = line_data(board, top_info)
        actual = line_data(board, played_info)
        top_score = max(top['score']['ordering_cp'], actual['score']['ordering_cp'])
        actual_score = actual['score']['ordering_cp']
        loss = top_score - actual_score
        if loss <= 50 and actual_score >= -100:
            brilliant = {'label': 'brilliant_candidate', 'detector': VERSION,
                         'evidence': evidence, 'confirmed_best': top, 'confirmed_played': actual,
                         'loss_cp': loss,
                         'reached_target_depth': min(top['depth'] or 0, actual['depth'] or 0) >= _SETTINGS['confirm_depth']}
    worst_score = ranked[-1]['score']['ordering_cp']
    return key, {'fen': board.fen(), 'side_to_move': 'white' if board.turn else 'black',
                 'legal_move_count': len(legal), 'evaluated_move_count': len(lines),
                 'minimum_depth': min(line['depth'] or 0 for line in ranked),
                 'best': ranked[0], 'played': played,
                 'worst_moves': [line for line in ranked if line['score']['ordering_cp'] == worst_score],
                 'worst_loss_cp': best_score - worst_score,
                 'legal_moves_ranked': ranked, 'brilliant': brilliant,
                 'method': 'finite_search_engine_estimate', 'settings': _SETTINGS}


def open_db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS config(value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS games(id TEXT PRIMARY KEY, status TEXT NOT NULL, reason TEXT, metadata TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS brilliant_keys(key TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS occurrences(
      game_id TEXT NOT NULL, ply INTEGER NOT NULL, position_key TEXT NOT NULL,
      is_user_move INTEGER NOT NULL, PRIMARY KEY(game_id,ply));
    CREATE INDEX IF NOT EXISTS occurrences_position ON occurrences(position_key);
    ''')
    return db


def load_games(source):
    seen = set()
    for path in sorted(source.glob('[0-9][0-9][0-9][0-9]-[0-9][0-9].json')):
        for game in json.loads(path.read_text()).get('games', []):
            identity = game.get('uuid') or game.get('url') or hashlib.sha256(json.dumps(game, sort_keys=True).encode()).hexdigest()
            if identity not in seen:
                seen.add(identity)
                yield identity, game


def mark_complete(db):
    with db:
        db.execute("""UPDATE games SET status='complete' WHERE status='pending' AND NOT EXISTS (
          SELECT 1 FROM occurrences o LEFT JOIN results r ON o.position_key=r.key
          WHERE o.game_id=games.id AND r.key IS NULL)""")


def write_progress(db, output, persist=True):
    statuses = dict(db.execute('SELECT status,count(*) FROM games GROUP BY status'))
    positions = db.execute('SELECT count(*) FROM results').fetchone()[0]
    brilliant = db.execute('SELECT count(*) FROM brilliant_keys').fetchone()[0]
    pending = db.execute('SELECT count(*) FROM occurrences o LEFT JOIN results r ON o.position_key=r.key WHERE r.key IS NULL').fetchone()[0]
    result = {'updated_at': time.time(), 'games': statuses, 'positions_analyzed': positions,
              'brilliant_candidate_positions': brilliant, 'pending_position_occurrences': pending,
              'complete': bool(statuses) and statuses.get('pending', 0) == 0 and pending == 0 and statuses.get('error', 0) == 0,
              'reporting_pid': os.getpid(), 'detector': VERSION}
    if persist:
        temporary = output / 'progress.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(output / 'progress.json')
    return result


def export(db, output):
    # Atomic snapshots: readers never see half-written files. IDs link to games.jsonl.
    names = ('positions', 'brilliant_moves', 'worst_moves')
    files = {name: (output / f'{name}.jsonl.tmp').open('w') for name in names}
    try:
        query = '''SELECT o.game_id,o.ply,o.is_user_move,r.payload FROM occurrences o
                   JOIN results r ON r.key=o.position_key ORDER BY o.game_id,o.ply'''
        for game_id, ply, is_user, payload in db.execute(query):
            data = json.loads(payload)
            context = {'game_id': game_id, 'ply': ply, 'is_user_move': bool(is_user),
                       'fen': data['fen'], 'side_to_move': data['side_to_move']}
            files['positions'].write(json.dumps(dict(context, **{k:v for k,v in data.items() if k not in context})) + '\n')
            files['worst_moves'].write(json.dumps(dict(context, moves=data['worst_moves'], best=data['best'],
                played=data['played'], loss_cp=data['worst_loss_cp'], minimum_depth=data['minimum_depth'],
                legal_move_count=data['legal_move_count'], evaluated_move_count=data['evaluated_move_count'],
                method=data['method'])) + '\n')
            if data['brilliant']:
                files['brilliant_moves'].write(json.dumps(dict(context, move=data['played'], **data['brilliant'])) + '\n')
    finally:
        for file in files.values():
            file.close()
    for name in names:
        (output / f'{name}.jsonl.tmp').replace(output / f'{name}.jsonl')
    with (output / 'games.jsonl.tmp').open('w') as file:
        for identity, status, reason, metadata in db.execute('SELECT * FROM games ORDER BY id'):
            file.write(json.dumps({'game_id': identity, 'status': status, 'reason': reason, 'metadata': json.loads(metadata)}) + '\n')
    (output / 'games.jsonl.tmp').replace(output / 'games.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', default='hydra33092')
    parser.add_argument('--source', type=Path, default=Path('data/players/hydra33092'))
    parser.add_argument('--output', type=Path, default=Path('data/analysis/hydra33092'))
    parser.add_argument('--engine', default=shutil.which('stockfish'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--seconds', type=float, default=2)
    parser.add_argument('--confirm-depth', type=int, default=18)
    parser.add_argument('--confirm-seconds', type=float, default=5)
    parser.add_argument('--max-positions', type=int)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    # Prevent overlapping writers, including accidental double-starts.
    import fcntl
    lock = (args.output / '.lock').open('w')
    if not args.status:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('An analyzer is already running for this output directory')
    db = open_db(args.output / 'analysis.sqlite3')
    if args.status:
        print(json.dumps(write_progress(db, args.output, persist=False), indent=2))
        return
    if args.export:
        export(db, args.output)
        return
    if not args.engine:
        parser.error('Stockfish executable not found')
    if min(args.workers, args.depth, args.confirm_depth) < 1 or min(args.seconds, args.confirm_seconds) <= 0:
        parser.error('Worker count and analysis budgets must be positive')
    with chess.engine.SimpleEngine.popen_uci(args.engine) as engine:
        engine_name = engine.id['name']
    settings = {'depth': args.depth, 'seconds': args.seconds, 'confirm_depth': args.confirm_depth,
                'confirm_seconds': args.confirm_seconds, 'engine': engine_name, 'detector': VERSION,
                'score_perspective': 'side_to_move_in_fen', 'threads_per_worker': 1}
    config = json.dumps(dict(settings, username=args.username.lower()), sort_keys=True)
    old = db.execute('SELECT value FROM config').fetchone()
    if old and old[0] != config:
        parser.error('Analysis configuration changed; use a new output directory')
    if not old:
        db.execute('INSERT INTO config VALUES (?)', (config,))
        db.commit()
    total = 0
    for identity, raw in load_games(args.source):
        total += 1
        if total % 500 == 0:
            logging.info('Indexed %s games', total)
        if db.execute('SELECT 1 FROM games WHERE id=?', (identity,)).fetchone():
            continue
        metadata = {k: v for k,v in raw.items() if k != 'pgn'}
        status, reason, positions = 'pending', None, []
        if not raw.get('pgn'):
            status, reason = 'skipped', 'missing_pgn'
        elif raw.get('rules') != 'chess':
            status, reason = 'skipped', 'unsupported_variant:' + str(raw.get('rules'))
        else:
            try:
                game = chess.pgn.read_game(io.StringIO(raw['pgn']))
                if game is None or game.errors or game.headers.get('Variant', 'Standard') not in ('Standard', 'Chess'):
                    raise ValueError('PGN parsing failed or unsupported variant')
                board = game.board()
                start = board.fen()
                history = []
                for ply, move in enumerate(game.mainline_moves(), 1):
                    if move not in board.legal_moves:
                        raise ValueError('Illegal move in PGN')
                    # History + played move prevents mixing draw contexts or classifications.
                    material = json.dumps([start, history, move.uci()], separators=(',', ':'))
                    key = hashlib.sha256(material.encode()).hexdigest()
                    side = 'white' if board.turn else 'black'
                    own = raw.get(side, {}).get('username', '').lower() == args.username.lower()
                    positions.append((identity, ply, key, int(own), start, list(history), move.uci()))
                    history.append(move.uci())
                    board.push(move)
                if not positions:
                    status, reason = 'skipped', 'no_played_moves'
            except (ValueError, IndexError) as exc:
                status, reason, positions = 'error', str(exc), []
        with db:
            db.execute('INSERT INTO games VALUES (?,?,?,?)', (identity, status, reason, json.dumps(metadata)))
            for game_id, ply, key, own, start, history, played in positions:
                db.execute('INSERT INTO occurrences VALUES (?,?,?,?)', (game_id, ply, key, own))
    if total == 0:
        parser.error('No monthly JSON games found in source directory')
    # Rebuild lightweight tasks from PGNs, preserving exact board history.
    def tasks():
        seen = set()
        for identity, raw in load_games(args.source):
            row = db.execute('SELECT status FROM games WHERE id=?', (identity,)).fetchone()
            if not row or row[0] != 'pending':
                continue
            game = chess.pgn.read_game(io.StringIO(raw['pgn']))
            start, history = game.board().fen(), []
            keys = dict(db.execute('SELECT ply,position_key FROM occurrences WHERE game_id=?', (identity,)))
            for ply, move in enumerate(game.mainline_moves(), 1):
                key = keys[ply]
                if key not in seen and not db.execute('SELECT 1 FROM results WHERE key=?', (key,)).fetchone():
                    seen.add(key)
                    yield key, start, list(history), move.uci()
                history.append(move.uci())
    write_progress(db, args.output)
    iterator = iter(tasks())
    completed, submitted, last_export = 0, 0, time.monotonic()
    started = last_progress = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
            initargs=(args.engine, settings)) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < args.workers * 2:
                if args.max_positions is not None and submitted >= args.max_positions:
                    exhausted = True
                    break
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                    break
                pending.add(pool.submit(analyze_position, task))
                submitted += 1
            if not pending:
                break
            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                key, data = future.result()
                with db:
                    db.execute('INSERT OR IGNORE INTO results VALUES (?,?)', (key, json.dumps(data)))
                    if data['brilliant']:
                        db.execute('INSERT OR IGNORE INTO brilliant_keys VALUES (?)', (key,))
                completed += 1
            if time.monotonic() - last_progress >= 30:
                progress = write_progress(db, args.output)
                logging.info('positions=%s new=%s seconds_per_position=%.2f candidates=%s',
                    progress['positions_analyzed'], completed, (time.monotonic()-started)/completed,
                    progress['brilliant_candidate_positions'])
                last_progress = time.monotonic()
            if time.monotonic() - last_export > 600:
                mark_complete(db)
                export(db, args.output)
                last_export = time.monotonic()
    mark_complete(db)
    export(db, args.output)
    print(json.dumps(write_progress(db, args.output), indent=2))


if __name__ == '__main__':
    main()
