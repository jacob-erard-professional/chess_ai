"""Non-neural baselines, scored like the nets (top-1 move match on target rows).

    python -m chessai.models.baselines --test data/datasets/hydra

frequency: play the move most often played in this exact position in a
    reference dataset (your training games first, then bulk games as fallback).
    Unseen positions count as misses; coverage is reported.
stockfish: play Stockfish's best move (rows with a joined eval only).
"""
import argparse
import json

import numpy as np

from ..dataset.reader import PositionDataset
from ..dataset.schema import NO_MOVE, SPLITS

_PRIME = np.uint64(1099511628211)


def position_keys(rows):
    """64-bit hash of (board, castling, en passant) in the mover's perspective."""
    words = np.ascontiguousarray(rows['pieces']).view(np.uint64).reshape(len(rows), 4)
    h = np.full(len(rows), 1469598103934665603, np.uint64)
    with np.errstate(over='ignore'):
        for col in (*words.T, rows['castling'].astype(np.uint64), rows['ep'].astype(np.uint64)):
            h = (h ^ col) * _PRIME
            h ^= h >> np.uint64(29)
    return h


class FrequencyTable:
    def __init__(self, keys, moves):
        order = np.lexsort((moves, keys))
        keys, moves = keys[order], moves[order]
        new_pair = np.r_[True, (keys[1:] != keys[:-1]) | (moves[1:] != moves[:-1])]
        starts = np.nonzero(new_pair)[0]
        counts = np.diff(np.r_[starts, len(keys)])
        pk, pm = keys[starts], moves[starts]
        best = np.lexsort((-counts, pk))  # per key, highest count first
        pk, pm = pk[best], pm[best]
        first = np.r_[True, pk[1:] != pk[:-1]]
        self.keys, self.moves = pk[first], pm[first]

    @classmethod
    def from_dataset(cls, ds, split='train', target=None):
        keys, moves = [], []
        for shard in ds.shards:
            keep = shard['split'] == SPLITS[split]
            if target is not None:
                keep &= shard['is_target'] == target
            rows = shard[keep]
            keys.append(position_keys(rows))
            moves.append(rows['move'].astype(np.uint16))
        return cls(np.concatenate(keys), np.concatenate(moves))

    def predict(self, keys):
        """(moves, found) for query keys."""
        pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        found = self.keys[pos] == keys
        return np.where(found, self.moves[pos], NO_MOVE), found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', default='data/datasets/hydra')
    ap.add_argument('--split', default='test')
    ap.add_argument('--bulk', default='data/datasets/bulk', help="fallback reference ('' to skip)")
    args = ap.parse_args()

    test = PositionDataset(args.test)
    idx = test.select(args.split, target=True if test.manifest.get('target_rows') else None)
    rows = test.rows(idx)
    keys, truth = position_keys(rows), rows['move']
    report = {'rows': len(rows)}

    own = FrequencyTable.from_dataset(test, 'train', target=True if test.manifest.get('target_rows') else None)
    pred, found = own.predict(keys)
    report['frequency_own'] = {'top1': float((pred == truth).mean()), 'coverage': float(found.mean())}
    if args.bulk:
        bulk = FrequencyTable.from_dataset(PositionDataset(args.bulk), 'train')
        bpred, bfound = bulk.predict(keys)
        combined = np.where(found, pred, bpred)
        report['frequency_bulk'] = {'top1': float((bpred == truth).mean()), 'coverage': float(bfound.mean())}
        report['frequency_own_then_bulk'] = {'top1': float((combined == truth).mean()),
                                             'coverage': float((found | bfound).mean())}
    has = rows['best_move'] != NO_MOVE
    report['stockfish_best'] = {'top1': float((rows['best_move'][has] == truth[has]).mean()) if has.any() else None,
                                'coverage': float(has.mean())}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
