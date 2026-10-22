#!/usr/bin/env python3
"""Resumable, serial Chess.com PubAPI dataset collector (stdlib only)."""
import argparse
import gzip
from email.utils import parsedate_to_datetime
import json
import logging
import random
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

API = 'https://api.chess.com/pub/'
KINDS = ('bullet', 'blitz', 'rapid')


def fetch(path, agent):
    for attempt in range(8):
        try:
            req = urllib.request.Request(API + path, headers={'User-Agent': agent, 'Accept-Encoding': 'gzip'})
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read()
                if response.headers.get('Content-Encoding') == 'gzip':
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return {}
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            retry = exc.headers.get('Retry-After', '')
            delay = min(300, 2 ** attempt * 3)
            if retry:
                try:
                    delay = max(delay, float(retry))
                except ValueError:
                    try:
                        delay = max(delay, parsedate_to_datetime(retry).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            delay = min(300, 2 ** attempt * 3)
        logging.warning('Retrying %s in %.1fs', path, delay)
        time.sleep(delay)
    raise RuntimeError('Retries exhausted: ' + path)


def connect(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS config (value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS jobs (path TEXT PRIMARY KEY, done INTEGER NOT NULL DEFAULT 0, priority INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS profile_stats(username TEXT PRIMARY KEY, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS profile_yield(username TEXT PRIMARY KEY, dry_streak INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS profiles (
        username TEXT PRIMARY KEY, game_count INTEGER NOT NULL, checked_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS game_sources (
        game_id TEXT NOT NULL, username TEXT NOT NULL, PRIMARY KEY(game_id,username));
    CREATE TABLE IF NOT EXISTS profile_policy (minimum INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS games (
        id TEXT PRIMARY KEY, bucket INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS games_bucket_kind ON games(bucket, kind);
    ''')
    if 'priority' not in {row[1] for row in db.execute('PRAGMA table_info(jobs)')}:
        db.execute('ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0')
    db.execute("""CREATE INDEX IF NOT EXISTS jobs_target_priority ON jobs(priority DESC,
      CASE WHEN path GLOB 'player/*/stats' THEN 0
           WHEN path GLOB 'player/*/games/[0-9]*' THEN 1 ELSE 2 END) WHERE done=0""")
    db.execute('DROP INDEX IF EXISTS jobs_pending_priority')
    db.commit()
    return db


def profile_game_count(stats):
    # Published stats record counts are rated results, not archive-month counts.
    return sum(stats.get('chess_' + kind, {}).get('record', {}).get(result, 0)
               for kind in KINDS for result in ('win', 'loss', 'draw'))


def target_priority(stats, current, quota):
    """Rank ratings near deficient average-rating buckets, allowing a 200-point margin."""
    best = 0
    for kind in KINDS:
        item = stats.get('chess_' + kind, {})
        rating = item.get('last', {}).get('rating')
        if not isinstance(rating, int):
            continue
        volume = sum(item.get('record', {}).get(r, 0) for r in ('win', 'loss', 'draw'))
        if volume == 0:
            continue
        for bucket in range(600, 3000, 100):
            deficit = max(0, quota - current.get((bucket, kind), 0))
            if deficit and bucket - 200 <= rating < bucket + 300:
                proximity = 1 if bucket <= rating < bucket + 100 else 0.6
                strength = deficit / quota * proximity * min(1, volume / 1000)
                best = max(best, max(1, int(strength * 1000)))
    return best


def prioritize_player(db, username, priority):
    prefix = f'player/{username}/'
    db.execute('UPDATE jobs SET priority=? WHERE done=0 AND path>=? AND path<?',
               (priority, prefix, prefix + '\uffff'))


def enqueue_job(db, path, priority=0):
    db.execute('''INSERT INTO jobs(path,priority) VALUES (?,?) ON CONFLICT(path)
      DO UPDATE SET priority=max(jobs.priority,excluded.priority) WHERE jobs.done=0''',
               (path, priority))


def profile_gate(db, path, minimum):
    """Return ready, pending (stats queued), or excluded for a player job."""
    parts = path.split('/')
    if len(parts) < 3 or parts[0] != 'player' or parts[2] != 'games':
        return 'ready'
    username = parts[1].lower()
    row = db.execute('SELECT game_count FROM profiles WHERE username=?', (username,)).fetchone()
    if row is None:
        priority = (db.execute('SELECT priority FROM jobs WHERE path=?', (path,)).fetchone() or [0])[0]
        enqueue_job(db, f'player/{username}/stats', priority + 1)
        db.commit()
        return 'pending'
    if row[0] < minimum:
        db.execute('UPDATE jobs SET done=1 WHERE substr(path,1,?)=?', (len(f'player/{username}/games/'), f'player/{username}/games/'))
        db.commit()
        return 'excluded'
    return 'ready'


def counts(db):
    return {(b, k): n for b, k, n in db.execute('''SELECT g.bucket,g.kind,count(*) FROM games g
        JOIN (SELECT DISTINCT game_id FROM game_sources) s ON s.game_id=g.id
        GROUP BY g.bucket,g.kind''')}


def report(db, output, quota, current=None):
    if current is None:
        current = counts(db)
    rows = [{'rating_min': b, 'rating_max': b + 99, 'time_class': k,
             'games': current.get((b, k), 0), 'target': quota}
            for b in range(600, 3000, 100) for k in KINDS]
    result = {'total': sum(current.values()), 'stored_including_unverified': db.execute('SELECT count(*) FROM games').fetchone()[0], 'target': 72 * quota,
              'complete': all(r['games'] == quota for r in rows), 'buckets': rows,
              'profile_filter': {'minimum_rated_games': (db.execute('SELECT minimum FROM profile_policy').fetchone() or [None])[0],
                  'checked_profiles': db.execute('SELECT count(*) FROM profiles').fetchone()[0],
                  'games_with_verified_source': sum(current.values()),
                  'pre_filter_games_are_unverified': True}}
    temp = output / 'progress.tmp'
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(output / 'progress.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data'))
    parser.add_argument('--quota', type=int, default=25000)
    parser.add_argument('--min-profile-games', type=int, default=1000)
    parser.add_argument('--user-agent', default='chess-ai-dataset/1.0')
    parser.add_argument('--seed', action='append', default=[])
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--max-requests', type=int)
    args = parser.parse_args()
    if args.min_profile_games < 1:
        parser.error('--min-profile-games must be positive')
    if args.quota < 1:
        parser.error('--quota must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    db = connect(args.output / 'games.sqlite3')
    config = json.dumps({'quota': args.quota, 'min': 600, 'max_exclusive': 3000, 'kinds': KINDS, 'bucket': 'floor((white+black)/200)*100'})
    old = db.execute('SELECT value FROM config').fetchone()
    if old and old[0] != config:
        parser.error('Existing dataset configuration differs; use another output directory')
    if not old:
        db.execute('INSERT INTO config VALUES (?)', (config,))
        db.commit()
    if args.status:
        print(json.dumps(report(db, args.output, args.quota), indent=2))
        return
    if args.export:
        for b in range(600, 3000, 100):
            for k in KINDS:
                with gzip.open(args.output / f'{b}-{b+99}_{k}.jsonl.gz', 'wt') as out:
                    for (payload,) in db.execute('SELECT payload FROM games WHERE bucket=? AND kind=? AND EXISTS (SELECT 1 FROM game_sources s WHERE s.game_id=games.id) ORDER BY id', (b, k)):
                        out.write(payload + '\n')
        return
    import fcntl
    lock = (args.output / '.collector.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A collector is already running for this output directory')
    policy = db.execute('SELECT minimum FROM profile_policy').fetchone()
    if policy and policy[0] != args.min_profile_games:
        parser.error('Existing profile threshold differs; use another output directory')
    if not policy:
        db.execute('INSERT INTO profile_policy VALUES (?)', (args.min_profile_games,))
        db.commit()
    def enqueue(path, priority=0):
        enqueue_job(db, path, priority)
    if not db.execute('SELECT 1 FROM jobs LIMIT 1').fetchone():
        enqueue('country/US/players')
        enqueue('titled/GM')
    for title in ('GM', 'IM', 'FM', 'NM'):
        enqueue(f'titled/{title}', 5000)
    enqueue('leaderboards', 5000)
    for seed in args.seed:
        enqueue(f'player/{seed.lower()}/games/archives', 2000)
    db.commit()
    requests = 0
    current = counts(db)
    last_report = 0
    while True:
        if time.monotonic() - last_report >= 30:
            report(db, args.output, args.quota, current)
            last_report = time.monotonic()
        if all(current.get((b, k), 0) >= args.quota for b in range(600, 3000, 100) for k in KINDS):
            logging.info('All quotas filled: %s games', sum(current.values()))
            break
        if args.max_requests is not None and requests >= args.max_requests:
            break
        job = db.execute("SELECT path,priority FROM jobs WHERE done=0 ORDER BY priority DESC, CASE WHEN path GLOB 'player/*/stats' THEN 0 WHEN path GLOB 'player/*/games/[0-9]*' THEN 1 ELSE 2 END, rowid LIMIT 1").fetchone()
        if not job:
            logging.warning('Frontier exhausted before quotas filled; add --seed USERNAME to expand')
            break
        path = job[0]
        if profile_gate(db, path, args.min_profile_games) != 'ready':
            continue
        parts = path.split('/')
        username = parts[1].lower() if parts[0] == 'player' else None
        if username and len(parts) > 2 and parts[2] == 'games':
            saved = db.execute('SELECT payload FROM profile_stats WHERE username=?', (username,)).fetchone()
            if saved is None:
                with db:
                    db.execute('''INSERT INTO jobs(path,priority) VALUES (?,?) ON CONFLICT(path)
                      DO UPDATE SET done=0,priority=excluded.priority''',
                               (f'player/{username}/stats', max(100, job[1] + 1)))
                continue
            strength = target_priority(json.loads(saved[0]), current, args.quota)
            if strength == 0 and job[1] >= 0:
                with db:
                    prioritize_player(db, username, -50)
                continue
        before = sum(current.values())
        data = fetch(path, args.user_agent)
        requests += 1
        with db:
            if path.endswith('/stats'):
                username = path.split('/')[1].lower()
                game_count = profile_game_count(data)
                db.execute('INSERT OR REPLACE INTO profile_stats VALUES (?,?)', (username, json.dumps(data)))
                strength = target_priority(data, current, args.quota)
                prioritize_player(db, username, 1000 + strength if strength else -50)
                db.execute('INSERT OR REPLACE INTO profiles VALUES (?,?,?)', (username, game_count, time.time()))
                if game_count < args.min_profile_games:
                    db.execute('UPDATE jobs SET done=1 WHERE substr(path,1,?)=?', (len(f'player/{username}/games/'), f'player/{username}/games/'))
                logging.info('Profile %s: %s rated games; eligible=%s', username, game_count, game_count >= args.min_profile_games)
            if 'players' in data:
                players = list(data['players'])
                random.Random(42).shuffle(players)
                for name in players:
                    enqueue(f'player/{name.lower()}/games/archives', 1000 if path.startswith('titled/') else 0)
            if path == 'leaderboards':
                for key in ('live_bullet', 'live_blitz', 'live_rapid'):
                    for player in data.get(key, []):
                        if player.get('username'):
                            enqueue(f"player/{player['username'].lower()}/games/archives", 1500)
            if 'archives' in data:
                # Visit all available history, newest first.
                for url in reversed(data['archives']):
                    if url.startswith(API):
                        enqueue(url[len(API):], job[1])
            for game in data.get('games', []):
                for side in ('white', 'black'):
                    name = game.get(side, {}).get('username')
                    if name:
                        kind_hint = game.get('time_class')
                        ratings = [game.get(color, {}).get('rating') for color in ('white', 'black')]
                        priority = 0
                        if kind_hint in KINDS and all(isinstance(r, int) for r in ratings):
                            bucket_hint = sum(ratings) // 200 * 100
                            if 600 <= bucket_hint < 3000 and current.get((bucket_hint, kind_hint), 0) < args.quota:
                                priority = 500
                        enqueue(f'player/{name.lower()}/games/archives', priority)
                kind = game.get('time_class')
                if kind not in KINDS or game.get('rules') != 'chess' or game.get('rated') is not True or not game.get('pgn'):
                    continue
                white = game.get('white', {}).get('rating')
                black = game.get('black', {}).get('rating')
                if not isinstance(white, int) or not isinstance(black, int):
                    continue
                bucket = ((white + black) // 200) * 100
                if not 600 <= bucket < 3000:
                    continue
                identity = game.get('uuid') or game.get('url')
                if not identity:
                    continue
                if current.get((bucket, kind), 0) >= args.quota:
                    continue
                was_verified = db.execute('SELECT 1 FROM game_sources WHERE game_id=? LIMIT 1', (identity,)).fetchone()
                db.execute('INSERT OR IGNORE INTO games VALUES (?,?,?,?)',
                                      (identity, bucket, kind, json.dumps(game, separators=(',', ':'))))
                db.execute('INSERT OR IGNORE INTO game_sources VALUES (?,?)', (identity, path.split('/')[1].lower()))
                current[bucket, kind] = current.get((bucket, kind), 0) + int(not was_verified)
            if username and len(parts) == 5 and parts[2] == 'games':
                added = sum(current.values()) - before
                db.execute('''INSERT INTO profile_yield VALUES (?,?) ON CONFLICT(username)
                  DO UPDATE SET dry_streak=CASE WHEN ? THEN 0 ELSE profile_yield.dry_streak+1 END''',
                           (username, 0 if added else 1, int(added > 0)))
                streak = db.execute('SELECT dry_streak FROM profile_yield WHERE username=?', (username,)).fetchone()[0]
                if streak >= 3:
                    prioritize_player(db, username, -100)
                    logging.info('Deferring low-yield history for %s after %s empty batches', username, streak)
            db.execute('UPDATE jobs SET done=1 WHERE path=?', (path,))
        logging.info('request=%s games=%s added=%s priority=%s source=%s', requests, sum(current.values()), sum(current.values())-before, job[1], path)
        time.sleep(0.3)
    report(db, args.output, args.quota, current)


if __name__ == '__main__':
    main()
