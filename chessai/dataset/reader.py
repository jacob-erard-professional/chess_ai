"""Read built datasets as numpy or torch batches.

    ds = PositionDataset('data/datasets/hydra')
    idx = ds.select('train', target=True)          # your moves in the training split
    for batch in ds.batches(idx, 1024, shuffle=True):
        batch['planes']    # (B, 17*(history+1), 8, 8) float32
        batch['features']  # (B, NUM_FEATURES) float32, see FEATURES
        batch['move']      # (B,) int64 target index into encoding.MOVES
        batch['legal']     # (B, 1858) bool, when the dataset has masks

`torch_batches` yields the same dicts as torch tensors on a device.
"""
import json
from pathlib import Path

import numpy as np

from .. import encoding
from .schema import NO_EVAL, ROW, SPLITS, unpack_pieces

FEATURES = ('elo', 'opp_elo', 'elo_diff', 'log_clock', 'log_opp_clock', 'clock_frac',
            'opp_clock_frac', 'log_base', 'inc', 'bullet', 'blitz', 'rapid', 'halfmove',
            'repetition', 'ply', 'timed')
NUM_FEATURES = len(FEATURES)


def features(rows):
    """Scalar conditioning features, roughly unit scale. Unknowns become 0."""
    f = np.zeros((len(rows), NUM_FEATURES), np.float32)
    elo, opp = rows['elo'].astype(np.float32), rows['opp_elo'].astype(np.float32)
    f[:, 0] = np.where(elo > 0, (elo - 1500) / 500, 0)
    f[:, 1] = np.where(opp > 0, (opp - 1500) / 500, 0)
    f[:, 2] = np.where((elo > 0) & (opp > 0), (elo - opp) / 200, 0)
    clock, opp_clock = rows['clock'], rows['opp_clock']
    timed = ~np.isnan(clock)
    base = np.maximum(rows['base'].astype(np.float32), 1)
    f[:, 3] = np.where(timed, np.log1p(np.nan_to_num(clock)) / 5, 0)
    f[:, 4] = np.where(timed, np.log1p(np.nan_to_num(opp_clock)) / 5, 0)
    f[:, 5] = np.where(timed, np.nan_to_num(clock) / base, 0)
    f[:, 6] = np.where(timed, np.nan_to_num(opp_clock) / base, 0)
    f[:, 7] = np.log1p(rows['base']) / 7
    f[:, 8] = rows['inc'] / 10
    for i in range(3):
        f[:, 9 + i] = rows['time_class'] == i
    f[:, 12] = rows['halfmove'] / 100
    f[:, 13] = (rows['repetition'].astype(np.float32) - 1) / 2
    f[:, 14] = rows['ply'] / 100
    f[:, 15] = timed
    return f


def _flip(pieces):
    """Board codes as seen by the other side: mirror ranks, swap colors."""
    flipped = pieces.reshape(-1, 8, 8)[:, ::-1].reshape(-1, 64)
    return np.where(flipped == 0, 0, np.where(flipped <= 6, flipped + 6, flipped - 6)).astype(np.uint8)


def stack_planes(rows, earlier):
    """Planes for `rows` followed by 12 piece planes per earlier position.

    earlier[k] = (rows k+1 plies back, bool valid mask). Earlier boards are
    re-oriented to the current mover; invalid ones become zero planes.
    """
    stack = [encoding.board_planes(unpack_pieces(rows['pieces']), rows['castling'], rows['ep'])]
    n = len(rows)
    for back, (prev, valid) in enumerate(earlier, 1):
        pieces = unpack_pieces(prev['pieces'])
        if back % 2:
            pieces = _flip(pieces)
        p = encoding.board_planes(pieces, np.zeros(n, np.uint8), np.full(n, encoding.NO_EP, np.uint8))[:, :12]
        p[~np.asarray(valid, bool)] = 0
        stack.append(p)
    return np.concatenate(stack, axis=1)


class PositionDataset:
    def __init__(self, root, in_memory=False):
        """in_memory=True reads every shard into RAM (bulk is ~10 GB); otherwise
        shards are memory-mapped and the first random pass is disk-bound."""
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest['encoding_version'] != encoding.ENCODING_VERSION:
            raise ValueError(f'{root} uses encoding v{self.manifest["encoding_version"]}, '
                             f'code is v{encoding.ENCODING_VERSION}; rebuild it')
        mode = None if in_memory else 'r'
        self.shards = [np.load(self.root / f"rows-{s['name']}.npy", mmap_mode=mode)
                       for s in self.manifest['shards']]
        self.legal = [np.load(self.root / f"legal-{s['name']}.npy", mmap_mode=mode)
                      for s in self.manifest['shards']] if self.manifest['legal_masks'] else None
        self.offsets = np.cumsum([0] + [len(s) for s in self.shards])
        self._games = None

    def __len__(self):
        return int(self.offsets[-1])

    @property
    def games(self):
        """Per-game metadata (list of dicts), loaded on first use."""
        if self._games is None:
            with open(self.root / 'games.jsonl') as f:
                self._games = [json.loads(line) for line in f]
        return self._games

    def column(self, name):
        return np.concatenate([s[name] for s in self.shards])

    def select(self, split=None, target=None, where=None):
        """Global row indices. where(rows) -> bool mask, applied per shard."""
        out = []
        for offset, shard in zip(self.offsets, self.shards):
            keep = np.ones(len(shard), bool)
            if split is not None:
                keep &= shard['split'] == SPLITS[split]
            if target is not None:
                keep &= shard['is_target'] == bool(target)
            if where is not None:
                keep &= where(shard)
            out.append(np.nonzero(keep)[0] + offset)
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    def rows(self, idx):
        """Structured rows (schema.ROW) for sorted or unsorted global indices."""
        idx = np.asarray(idx)
        out = np.empty(len(idx), ROW)
        shard_of = np.searchsorted(self.offsets, idx, side='right') - 1
        for s in np.unique(shard_of):
            sel = shard_of == s
            out[sel] = self.shards[s][idx[sel] - self.offsets[s]]
        return out

    def legal_masks(self, idx):
        idx = np.asarray(idx)
        out = np.empty((len(idx), self.legal[0].shape[1]), np.uint8)
        shard_of = np.searchsorted(self.offsets, idx, side='right') - 1
        for s in np.unique(shard_of):
            sel = shard_of == s
            out[sel] = self.legal[s][idx[sel] - self.offsets[s]]
        return np.unpackbits(out, axis=1, count=encoding.NUM_MOVES).astype(bool)

    def planes(self, idx, history=0):
        """Board planes for rows idx, plus `history` earlier positions of the same game.

        Earlier positions are re-oriented to the current mover; positions
        before the game started are left as zero planes.
        """
        idx = np.asarray(idx)
        rows = self.rows(idx)
        earlier = []
        for back in range(1, history + 1):
            prev = self.rows(np.maximum(idx - back, 0))
            valid = (prev['game'] == rows['game']) & (prev['ply'] == rows['ply'] - back) & (idx >= back)
            earlier.append((prev, valid))
        return stack_planes(rows, earlier)

    def batch(self, idx, history=0):
        idx = np.sort(np.asarray(idx))  # sorted reads are much faster on memmaps
        rows = self.rows(idx)
        out = {'index': idx, 'planes': self.planes(idx, history), 'features': features(rows),
               'move': rows['move'].astype(np.int64), 'result': rows['result'].astype(np.float32),
               'move_time': rows['move_time'], 'is_target': rows['is_target'].astype(bool),
               'eval_cp': rows['eval_cp'], 'has_eval': rows['eval_cp'] != NO_EVAL,
               'game': rows['game'], 'ply': rows['ply']}
        if self.legal is not None:
            out['legal'] = self.legal_masks(idx)
        return out

    def batches(self, idx, batch_size, shuffle=False, seed=0, history=0, drop_last=False):
        idx = np.asarray(idx)
        if shuffle:
            idx = np.random.default_rng(seed).permutation(idx)
        stop = len(idx) - (len(idx) % batch_size if drop_last else 0)
        for i in range(0, stop, batch_size):
            yield self.batch(idx[i:i + batch_size], history)

    def torch_batches(self, idx, batch_size, device=None, **kw):
        import torch
        device = device or default_device()
        for b in self.batches(idx, batch_size, **kw):
            yield {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in b.items()}


def default_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')
