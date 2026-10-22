"""Build a training dataset of positions from chess.com-format games.

    # your games: moves by Hydra33092 flagged is_target; time split (newest 10% test)
    python -m chessai.dataset.build archive data/players/hydra33092 \\
        --target hydra33092 --out data/datasets/hydra --legal

    # the bulk rating dataset (read-only; safe while the collector runs)
    python -m chessai.dataset.build bulk data/games.sqlite3 \\
        --exclude hydra33092 --out data/datasets/bulk

Output (`--out`):
    manifest.json      row dtype, counts, sources, skip reasons, encoding version
    games.jsonl        one line per game; rows[...]['game'] is the line number
    rows-NNNNN.npy     schema.ROW arrays, load with np.load(mmap_mode='r')
    legal-NNNNN.npy    (with --legal) np.packbits legal-move masks, aligned with rows
"""
import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

from .. import encoding
from .extract import Skip, extract, game_meta, hash_split
from .schema import ROW, TEST, TRAIN, VAL

CHUNK_GAMES = 2000
ARCHIVE_CHUNK_GAMES = 50


class ShardWriter:
    def __init__(self, out, shard_rows, legal):
        self.out, self.shard_rows, self.legal = Path(out), shard_rows, legal
        self.out.mkdir(parents=True)
        self.games = open(self.out / 'games.jsonl', 'w')
        self.rows, self.masks, self.buffered = [], [], 0
        self.shards, self.total_rows, self.total_games = [], 0, 0
        self.split_rows = collections.Counter()
        self.target_rows = 0

    def add(self, metas, rows, masks):
        """metas: per-game dicts; rows['game'] holds 0-based indices into metas."""
        rows['game'] += self.total_games
        for meta in metas:
            self.games.write(json.dumps(meta) + '\n')
        self.total_games += len(metas)
        self.rows.append(rows)
        if self.legal:
            self.masks.append(masks)
        self.buffered += len(rows)
        if self.buffered >= self.shard_rows:
            self.flush()

    def flush(self):
        if not self.buffered:
            return
        rows = np.concatenate(self.rows)
        name = f'{len(self.shards):05d}'
        np.save(self.out / f'rows-{name}.npy', rows)
        if self.legal:
            np.save(self.out / f'legal-{name}.npy', np.concatenate(self.masks))
        self.shards.append({'name': name, 'rows': len(rows)})
        self.total_rows += len(rows)
        self.target_rows += int(rows['is_target'].sum())
        for split, count in zip(*np.unique(rows['split'], return_counts=True)):
            self.split_rows[int(split)] += int(count)
        self.rows, self.masks, self.buffered = [], [], 0

    def close(self, manifest):
        self.flush()
        self.games.close()
        names = {TRAIN: 'train', VAL: 'val', TEST: 'test'}
        manifest.update({
            'encoding_version': encoding.ENCODING_VERSION, 'num_moves': encoding.NUM_MOVES,
            'row_dtype': ROW.descr, 'legal_masks': self.legal,
            'games': self.total_games, 'rows': self.total_rows, 'target_rows': self.target_rows,
            'split_rows': {names[k]: v for k, v in sorted(self.split_rows.items())},
            'shards': self.shards,
        })
        (self.out / 'manifest.json').write_text(json.dumps(manifest, indent=1))


def _extract_chunk(games, targets, legal):
    """games: [(data, split)] -> (metas, rows, masks, skip counts)."""
    metas, rows, masks, skips = [], [], [], collections.Counter()
    for data, split in games:
        try:
            r, m = extract(data, len(metas), split, targets, legal)
        except Skip as e:
            skips[(_reason(e), data.get('url') or data.get('uuid'))] += 1
            continue
        meta = game_meta(data)
        meta['split'] = split
        meta['plies'] = len(r)
        metas.append(meta)
        rows.append(r)
        masks.append(m)
    if not rows:
        return metas, np.zeros(0, ROW), None, skips
    return metas, np.concatenate(rows), np.concatenate(masks) if legal else None, skips


def _reason(error):
    return str(error).split(':')[0].split(' at ply')[0]


class SkipLog:
    """Counts per reason, plus a few example games for each."""

    def __init__(self, examples=5):
        self.counts, self.examples, self.limit = collections.Counter(), {}, examples

    def update(self, skips):
        for (reason, game), n in skips.items():
            self.counts[reason] += n
            ex = self.examples.setdefault(reason, [])
            if game and len(ex) < self.limit:
                ex.append(game)

    def summary(self):
        return {r: {'games': n, 'examples': self.examples.get(r, [])} for r, n in self.counts.most_common()}


# --- archive source: monthly {"games": [...]} JSON files ---------------------
def load_archive(paths):
    games, seen = [], set()
    for path in paths:
        files = sorted(glob.glob(str(Path(path) / '[0-9]*.json'))) if Path(path).is_dir() else [path]
        for f in files:
            with open(f) as fh:
                archive = json.load(fh)['games']
            for g in archive:
                key = g.get('uuid') or g.get('url')
                if key not in seen:
                    seen.add(key)
                    games.append(g)
    games.sort(key=lambda g: g.get('end_time') or 0)
    return games


def time_splits(n, val=5, test=10):
    """Oldest games train, then val, newest test (percentages)."""
    test_start, val_start = n - n * test // 100, n - n * (test + val) // 100
    return [TRAIN if i < val_start else VAL if i < test_start else TEST for i in range(n)]


def _archive_worker(args):
    return _extract_chunk(*args)


def build_archive(args, writer):
    games = load_archive(args.sources)
    splits = time_splits(len(games)) if args.split == 'time' else \
        [hash_split(g.get('uuid') or g['url']) for g in games]
    pairs = list(zip(games, splits))
    chunks = [(pairs[i:i + ARCHIVE_CHUNK_GAMES], args.targets, args.legal)
              for i in range(0, len(pairs), ARCHIVE_CHUNK_GAMES)]
    skips = SkipLog()
    with mp.Pool(args.workers) as pool:
        for metas, rows, masks, s in pool.imap(_archive_worker, chunks):
            writer.add(metas, rows, masks)
            skips.update(s)
    return {'source': 'archive', 'paths': [str(p) for p in args.sources], 'split_mode': args.split}, skips


# --- bulk source: the collector's SQLite database ----------------------------
_DB = None


def _bulk_init(db_path):
    global _DB
    _DB = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)


def _bulk_worker(job):
    lo, hi, verified, exclude, targets, legal = job
    query = 'SELECT payload FROM games g WHERE rowid BETWEEN ? AND ?'
    if verified:
        query += ' AND EXISTS (SELECT 1 FROM game_sources s WHERE s.game_id=g.id)'
    games, skipped = [], collections.Counter()
    for (payload,) in _DB.execute(query + ' ORDER BY rowid', (lo, hi)):
        data = json.loads(payload)
        if {data['white']['username'].lower(), data['black']['username'].lower()} & exclude:
            skipped[('excluded player', None)] += 1
            continue
        games.append((data, hash_split(data.get('uuid') or data['url'])))
    metas, rows, masks, skips = _extract_chunk(games, targets, legal)
    skips.update(skipped)
    return metas, rows, masks, skips


def build_bulk(args, writer):
    db_path = Path(args.sources[0]).resolve()
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    max_rowid = con.execute('SELECT max(rowid) FROM games').fetchone()[0] or 0
    if args.limit:
        max_rowid = min(max_rowid, args.limit)
    con.close()
    jobs = [(lo, min(lo + CHUNK_GAMES - 1, max_rowid), args.verified_only, args.exclude,
             args.targets, args.legal) for lo in range(1, max_rowid + 1, CHUNK_GAMES)]
    skips, started = SkipLog(), time.time()
    with mp.Pool(args.workers, _bulk_init, (str(db_path),)) as pool:
        for i, (metas, rows, masks, s) in enumerate(pool.imap(_bulk_worker, jobs), 1):
            writer.add(metas, rows, masks)
            skips.update(s)
            if i % 50 == 0 or i == len(jobs):
                rate = writer.total_games / (time.time() - started)
                print(f'{i}/{len(jobs)} chunks, {writer.total_games:,} games, '
                      f'{writer.total_rows + writer.buffered:,} rows, {rate:,.0f} games/s', flush=True)
    return {'source': 'bulk', 'db': str(db_path), 'max_rowid': max_rowid,
            'verified_only': args.verified_only, 'excluded': sorted(args.exclude),
            'split_mode': 'hash'}, skips


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=('archive', 'bulk'))
    ap.add_argument('sources', nargs='+', type=Path,
                    help='archive: directories/files of monthly JSON; bulk: the SQLite db')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--target', action='append', default=[], help='username whose moves are is_target')
    ap.add_argument('--exclude', action='append', default=[], help='bulk: drop games with this player')
    ap.add_argument('--legal', action='store_true', help='also store legal-move masks (233 B/row)')
    ap.add_argument('--split', choices=('time', 'hash'), default='time', help='archive split mode')
    ap.add_argument('--verified-only', action='store_true', help='bulk: only profile-verified games')
    ap.add_argument('--limit', type=int, help='bulk: only rowids up to N (for quick trials)')
    ap.add_argument('--shard-rows', type=int, default=2_000_000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--force', action='store_true', help='replace an existing --out')
    args = ap.parse_args()
    args.targets = frozenset(t.lower() for t in args.target)
    args.exclude = frozenset(t.lower() for t in args.exclude)

    if args.out.exists() and not args.force:
        ap.error(f'{args.out} exists (use --force to replace it)')
    tmp = args.out.with_name(args.out.name + '.building')
    shutil.rmtree(tmp, ignore_errors=True)
    started = time.time()
    writer = ShardWriter(tmp, args.shard_rows, args.legal)
    info, skips = (build_archive if args.kind == 'archive' else build_bulk)(args, writer)
    info.update({'targets': sorted(args.targets), 'built_at': time.time(),
                 'build_seconds': round(time.time() - started, 1), 'skipped': skips.summary()})
    writer.close(info)
    if args.out.exists():
        shutil.rmtree(args.out)
    os.replace(tmp, args.out)
    print(f'{writer.total_games:,} games, {writer.total_rows:,} rows '
          f'({writer.target_rows:,} target) -> {args.out} in {info["build_seconds"]}s; skipped {dict(skips.counts)}')


if __name__ == '__main__':
    main()
