#!/usr/bin/env python3
"""Download every available monthly archive for one Chess.com player."""
import json
import sys
import time
from pathlib import Path
from collect_games import fetch

username = sys.argv[1].lower()
if not username.replace('_', '').replace('-', '').isalnum():
    raise SystemExit('Invalid username')
output = Path('data/players') / username
output.mkdir(parents=True, exist_ok=True)
agent = 'chess-ai-dataset/1.0'
archives = fetch(f'player/{username}/games/archives', agent).get('archives', [])
if not archives:
    raise SystemExit('No archives available for ' + username)
(output / 'archives.json').write_text(json.dumps(archives, indent=2) + '\n')
total = 0
for url in archives:
    year, month = url.rstrip('/').split('/')[-2:]
    target = output / f'{year}-{month}.json'
    if target.exists():
        data = json.loads(target.read_text())
    else:
        time.sleep(2)
        data = fetch(f'player/{username}/games/{year}/{month}', agent)
        if 'games' not in data:
            raise RuntimeError('Archive unavailable: ' + url)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(data) + '\n')
        temporary.replace(target)
    total += len(data['games'])
    print(f'{year}-{month}: {len(data["games"])} games; total={total}', flush=True)
with (output / f'{username}.pgn').open('w') as pgn:
    for url in archives:
        year, month = url.rstrip('/').split('/')[-2:]
        data = json.loads((output / f'{year}-{month}.json').read_text())
        for game in data['games']:
            if game.get('pgn'):
                pgn.write(game['pgn'].strip() + '\n\n')
(output / 'summary.json').write_text(json.dumps({'username': username, 'games': total, 'months': len(archives)}, indent=2) + '\n')
print(f'Complete: {total} games in {output}', flush=True)
