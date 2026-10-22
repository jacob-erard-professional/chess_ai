"""Join Stockfish analysis (positions.jsonl from scripts/analyze_games.py) into a dataset.

    python -m chessai.dataset.evals data/datasets/hydra data/analysis/hydra33092/positions.jsonl

Fills eval_cp, played_cp and best_move in place. Safe to rerun as the analysis
progresses; rows without analysis keep NO_EVAL / NO_MOVE. Each joined row is
checked against the analysed FEN and played move, so a mismatched analysis
file cannot silently corrupt labels.
"""
import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np

from .. import encoding
from .reader import PositionDataset
from .schema import EVAL_CLIP, pack_pieces


def score_cp(score):
    """Analysis score (mover's view) -> int16 centipawns; mate -> +-(EVAL_CLIP - distance)."""
    if score.get('mate') is not None:
        mate = score['mate']
        return int(np.sign(mate) * (EVAL_CLIP - min(abs(mate), 999)))
    return int(np.clip(score['cp'], -(EVAL_CLIP - 1000), EVAL_CLIP - 1000))


def join(root, positions_path):
    ds = PositionDataset(root)
    uuid_to_game = {g['uuid']: i for i, g in enumerate(ds.games)}
    game_col = ds.column('game')
    first_row = np.searchsorted(game_col, np.arange(len(ds.games)))
    shards = [np.load(Path(root) / f"rows-{s['name']}.npy", mmap_mode='r+') for s in ds.manifest['shards']]
    stats = dict(lines=0, joined=0, unknown_game=0, mismatched=0)
    with open(positions_path) as f:
        for line in f:
            stats['lines'] += 1
            p = json.loads(line)
            game = uuid_to_game.get(p['game_id'])
            if game is None:
                stats['unknown_game'] += 1
                continue
            index = int(first_row[game]) + p['ply'] - 1
            s = np.searchsorted(ds.offsets, index, side='right') - 1
            row = shards[s][index - ds.offsets[s]]
            board = chess.Board(p['fen'])
            pieces, _, _ = encoding.encode_board(board)
            played = encoding.move_to_index(chess.Move.from_uci(p['played']['uci']), board.turn)
            if (row['game'] != game or row['ply'] != p['ply'] - 1 or row['move'] != played
                    or not np.array_equal(row['pieces'], pack_pieces(pieces)[0])):
                stats['mismatched'] += 1
                continue
            row['eval_cp'] = score_cp(p['best']['score'])
            row['played_cp'] = score_cp(p['played']['score'])
            row['best_move'] = encoding.move_to_index(chess.Move.from_uci(p['best']['uci']), board.turn)
            stats['joined'] += 1
    for shard in shards:
        shard.flush()
    manifest = ds.manifest
    manifest['evals'] = {'source': str(positions_path), 'joined_at': time.time(), **stats}
    (Path(root) / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', type=Path)
    ap.add_argument('positions', type=Path)
    args = ap.parse_args()
    print(join(args.dataset, args.positions))


if __name__ == '__main__':
    main()
