#!/usr/bin/env python3
"""Reprioritize saved jobs toward exact deficient rating buckets without API calls."""
import datetime as dt
import fcntl
import json
from pathlib import Path
from collect_games import connect, counts, prioritize_player, report, KINDS


def retarget(db, current, quota=25000, minimum=1000):
    targeted = []
    profiles = db.execute('''SELECT s.username,s.payload FROM profile_stats s
        JOIN profiles p ON p.username=s.username WHERE p.game_count>=?''', (minimum,)).fetchall()
    with db:
        for username, payload in profiles:
            stats = json.loads(payload)
            choices = []
            for kind in KINDS:
                item = stats.get('chess_' + kind, {})
                rating = item.get('last', {}).get('rating')
                date = item.get('last', {}).get('date')
                if not isinstance(rating, int):
                    continue
                bucket = rating // 100 * 100
                deficit = quota - current.get((bucket, kind), 0)
                if not 600 <= bucket < 3000 or deficit <= 0:
                    continue
                volume = sum(item.get('record', {}).get(r, 0) for r in ('win', 'loss', 'draw'))
                if volume:
                    choices.append((deficit, min(volume, 10000), kind, date, bucket))
            if not choices:
                continue
            deficit, volume, kind, date, bucket = max(choices)
            prefix = f'player/{username}/games/'
            jobs = db.execute('SELECT path FROM jobs WHERE done=0 AND path>=? AND path<?',
                              (prefix, prefix+'\uffff')).fetchall()
            if not jobs:
                continue
            # Exact deficit matches lead the ordinary discovery queue. No jobs
            # are reopened or deleted; only unprocessed history is revisited.
            priority = 10000 + int(deficit / quota * 1000)
            prioritize_player(db, username, priority)
            if isinstance(date, (int, float)):
                last = dt.datetime.fromtimestamp(date, dt.timezone.utc)
                last_month = last.year * 12 + last.month
                for (path,) in jobs:
                    parts = path.split('/')
                    if len(parts) == 5 and parts[-2].isdigit() and parts[-1].isdigit():
                        distance = abs(int(parts[-2]) * 12 + int(parts[-1]) - last_month)
                        db.execute('UPDATE jobs SET priority=? WHERE path=? AND done=0',
                                   (priority + max(0, 60 - distance), path))
            db.execute('DELETE FROM profile_yield WHERE username=?', (username,))
            targeted.append({'username': username, 'time_class': kind, 'rating_bucket': bucket,
                             'deficit': deficit, 'pending_jobs': len(jobs), 'priority': priority})
    return targeted


if __name__ == '__main__':
    output = Path('data')
    with (output / '.collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = connect(output / 'games.sqlite3')
        current = counts(db)
        targeted = retarget(db, current)
        (output / 'retargeted_profiles.json').write_text(json.dumps(targeted, indent=2)+'\n')
        report(db, output, 25000, current)
        print(json.dumps({'verified_games': sum(current.values()), 'targeted_profiles': len(targeted),
                          'targeted_jobs': sum(p['pending_jobs'] for p in targeted)}, indent=2))
