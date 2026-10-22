# Chess.com rating dataset

Target: 25,000 unique rated standard games for each of bullet, blitz, and rapid in each 100-point average-rating interval from 600–699 through 2900–2999 (1,800,000 games).

The bucket uses the ratings stored in the game, not current player ratings. Each game belongs to one bucket. Full API game JSON includes PGN, players, results, ratings, time control, and source URL.

Run commands from the repository root.

```sh
python3 scripts/collect_games.py
python3 scripts/collect_games.py --status
python3 scripts/collect_games.py --export
```

Data lives in `data/games.sqlite3`. `data/progress.json` reports every quota. Export creates 72 compressed JSONL files. Resume by running the same command; completed requests and duplicate games are skipped. Stop with SIGTERM or Ctrl-C; each processed response commits atomically. Run only one collector per output directory.

Requests are serial, with a 0.3-second pause after each completed request (at most 200 requests/minute), with backoff for transient failures and rate limiting. Use `--user-agent 'your-app (contact@example.com)'` to supply contact information. Optional `--seed USERNAME` adds player discovery seeds; `--max-requests N` bounds a run.

Discovery starts with the public US player list and GM roster, then expands through opponents and player archives. This is a convenience sample, **not a random or representative sample** of Chess.com. All available months are eligible, newest first for each discovered player. Rare high-rating buckets may not fill. The collector reports deficits rather than duplicating games or loosening filters. API failures leave the current request pending for restart.

API reference: https://www.chess.com/news/view/published-data-api

## Hydra33092 move analysis

Stockfish runs locally; this does not call Chess.com analysis servers. Install dependencies with `python3 -m venv .venv`, `.venv/bin/pip install -r requirements-analysis.txt`, and `brew install stockfish` on macOS.

```sh
.venv/bin/python scripts/analyze_games.py
.venv/bin/python scripts/analyze_games.py --status
.venv/bin/python scripts/analyze_games.py --export
```

The analyzer reads the monthly JSON in `data/players/hydra33092` and writes `data/analysis/hydra33092/analysis.sqlite3`. It checkpoints every position and resumes with the same command. A file lock prevents overlapping analysis writers. It uses two worker processes, each with one engine thread and 128 MB hash. Default screening searches all legal moves with MultiPV, targeting depth 12 with a 2-second limit per position. Missing root lines receive explicit forced searches; incomplete coverage is never committed. Candidate confirmation targets depth 18 with up to 5 seconds each for unrestricted and forced-played searches. The engine can hit a time limit before the requested depth; actual depths are saved. Settings cannot change in an existing output directory.

This is a substantial offline job over every played position, including both players. Full game histories are passed to the engine so repetition context is preserved. Duplicate game/position occurrences link to shared analysis where the history and played move match. Missing PGNs, unsupported variants, and parsing failures are explicitly recorded in the games manifest. Parsing failures prevent a complete status. `progress.json` reports totals; completed game statuses and JSON exports refresh approximately every ten minutes and at normal exit. Individual position results are immediately available in SQLite. Stop and restart without losing committed positions.

Website exports (JSONL: one JSON object per line):

- `games.jsonl`: game ID, source URL, player metadata, status and skip/error reason.
- `positions.jsonl`: game ID, one-based ply, FEN, `is_user_move`, played move, full legal-move ranking, and analysis settings.
- `brilliant_moves.jsonl`: **played brilliant candidates**, with material evidence and deeper verification scores/depths. Filter `is_user_move: true` for Hydra33092's own moves.
- `worst_moves.jsonl`: lowest-scoring legal move(s) in every analyzed position, plus best/played alternatives and score loss. These are hypothetical alternatives, not necessarily moves anyone played. Ties are retained.

Moves include UCI, SAN, evaluation, actual depth and UCI principal variation. Scores are always from the perspective of the player to move in the exported FEN: positive favors that player, negative favors their opponent. `cp` is null for mate scores; `mate` is signed and null for centipawn scores. `ordering_cp` maps mate scores near ±100,000 for sorting; it is not a literal centipawn evaluation. Worst move rankings are finite-search estimates, not mathematical proofs. Shallow results should be reanalyzed before using them as definitive puzzles.

Brilliant candidates require a moved knight/bishop/rook/queen to offer at least a pawn of net material in a sequence of **legal captures** on its destination. Material already captured by the move is subtracted. Pinned attackers, defended equal exchanges, king moves and ordinary pawn moves do not qualify. Deeper confirmation requires at most 50cp loss versus the best found continuation and a position no worse than -100cp. `reached_target_depth` distinguishes full-depth confirmation from time-limited confirmation. This is a documented heuristic, not Chess.com's classifier. It can miss positional sacrifices, sacrifices of pieces other than the one moved, and other human notions of brilliance. All labels remain `brilliant_candidate`.

Run verification: `.venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v`.

## High-volume profile filter

The rating-dataset collector checks `/player/{username}/stats` before requesting any new player archive. It requires at least **1,000 lifetime rated bullet + blitz + rapid games combined**, summing wins/losses/draws. Daily games, variants and puzzles do not count. Override with `--min-profile-games N` in a new output directory. A qualifying source player is required; their opponent can have fewer games.

The filter applies to future extraction. The 228,982 previously collected games are retained and are unverified unless subsequently associated with a qualifying source. `profiles` stores player counts and check timestamps; `game_sources` links games to qualifying source players. `progress.json` reports how many games have a verified source. To query only verified games, use `SELECT g.* FROM games g WHERE EXISTS (SELECT 1 FROM game_sources s WHERE s.game_id=g.id)`. Quota progress and standard bulk exports include only games with verified qualifying sources; older unverified games remain stored separately in the same database and do not count toward the quotas. A pre-filter snapshot is saved in `data/before-profile-filter.sqlite3`. The personal Hydra33092 archive and analysis are independent of this filter.

Collector performance: quota counts are rebuilt from SQLite at startup, then maintained in memory. Progress snapshots refresh every 30 seconds and at normal exit. The pending-job queue uses a partial priority index. Only one collector can own an output directory at a time. Serial requests and the 0.3-second pause are unchanged.

## Rating-aware scheduling

The collector prioritizes profiles whose latest bullet/blitz/rapid rating is near an unfilled bucket for that same time class. Larger deficits and higher game volume receive greater priority. A 200-point margin accommodates opponent ratings and rating movement; final inclusion still uses the actual average of both ratings in each game. Current ratings are only a scheduling heuristic and cannot guarantee the ratings of historical games.

Titled-player lists (GM/IM/FM/NM) and live leaderboards seed discovery for high-rating buckets. Opponents observed in deficient buckets receive a discovery boost. Cached full player stats guide later scheduling. Profiles still require 1,000 rated games combined. Three consecutive archive batches adding no qualifying unique games defer that player's remaining history. Deferred jobs remain available rather than being deleted or marked completed. Profiles whose current ratings no longer match a deficit are also deferred; they can be revisited after higher-priority discovery is exhausted.

The existing pending queue is migrated in place to add an indexed priority. Saved games, quotas, deduplication, and the 0.3-second serial throttle remain unchanged. Logs include `added` and `priority` per response. Hydra's download and analysis are unaffected.

To retarget already-cached profiles after many quotas fill, stop the collector and run `python3 scripts/retarget_downloads.py`, then restart it. The command takes the collector lock, preserves all completed jobs and saved games, boosts qualifying profiles whose current rating is inside an exact missing bucket, and prioritizes pending months closest to that time control's last-rated-game date. It resets those profiles' dry streaks for one more pass over unprocessed history. The ordinary three-empty-batch deferral remains active. `data/retargeted_profiles.json` records the selected profiles and targets.

## Stockfish on the PC

The `pc` SSH host runs analysis from `~/projects/chess_ai` using Stockfish 19 and eight single-threaded workers. The user service in `scripts/stockfish-analysis.service` resumes the transferred SQLite checkpoint; the engine version and search budgets match the laptop run. Inputs and checkpoints live under `data/` and are transferred over SSH, not committed to Git. Website exports are rebuilt on the PC. The game downloader remains on the laptop.

```sh
ssh pc 'systemctl --user status stockfish-analysis.service'
ssh pc 'journalctl --user -u stockfish-analysis.service -n 30 --no-pager'
ssh pc 'cat ~/projects/chess_ai/data/analysis/hydra33092/progress.json'
ssh pc 'systemctl --user stop stockfish-analysis.service'
ssh pc 'systemctl --user start stockfish-analysis.service'
```

The enabled user service survives SSH disconnects and starts after PC reboots when user lingering is enabled. Laptop and PC checkpoints are separate copies after the handoff; keep only the PC analysis running. To retrieve exported results after a checkpoint/export cycle:

```sh
rsync -a --include='*.jsonl' --include='progress.json' --exclude='*' pc:projects/chess_ai/data/analysis/hydra33092/ ./data/analysis/hydra33092/
```

Stop the remote service before copying the live SQLite database back. Copying an active WAL database without its current WAL is not a consistent snapshot. Individual JSONL exports are atomically replaced on the PC and can be retrieved while analysis continues.

## Engine and training data (`chessai/`)

Setup: `uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt` (`brew install stockfish` for the Stockfish player). Tests: `.venv/bin/python -m unittest discover -s tests -t .`

**Engine** (`chessai.engine`). `Game` applies every standard rule (python-chess move generation; castling, en passant, under/promotion) plus the game layer: clocks with increment, flagging (`timevsinsufficient` when the opponent cannot mate), resignation, agreement, abandonment, stalemate, insufficient material (chess.com rule: also KB v KN, KN v KN, KB v KB and KNN v K), and repetition / 50-move draws. `auto_draw=True` (default) ends threefold and 50-move draws immediately, like chess.com; `auto_draw=False` requires `claim_draw()` and only fivefold / 75-move are automatic (FIDE). `Game.to_chesscom()` emits the same JSON as `data/` (PGN with `%clk`, TCN, result codes, termination text); `Game.from_chesscom()` replays any stored game. All 12,786 standard Hydra33092 games reproduce their PGN moves, clocks, TCN, results and termination exactly.

Players implement `choose_move(game) -> Move | RESIGN` and see the whole game (clocks, ratings, history): `human`, `random`, `alphabeta` (negamax, iterative deepening, quiescence, piece-square eval with endgame mop-up; `chessai/engine/search.py`; `time=` seconds per move or ~1/30 of the clock when timed, deepening until time runs out, or a fixed `depth=`, default 4 when untimed), `stockfish` (optionally `elo=` 1320–3190).

```sh
.venv/bin/python -m chessai.engine.play --white human --black alphabeta --tc 180+2
.venv/bin/python -m chessai.engine.play --white stockfish:elo=1500 --black alphabeta:time=0.5 --games 20 --swap
```

Games are appended to `data/played/YYYY-MM.json` in the monthly-archive layout, so they feed the dataset builder unchanged.

**Encoding** (`chessai.encoding`, the single source for every model). Positions are from the side to move's view (Black's positions are mirrored and colors swapped). Board: 64 piece codes → 17 planes (12 pieces, 4 castling, en passant). Moves: a fixed 1,858-entry vocabulary (queen/knight-reachable from→to plus underpromotions). `ENCODING_VERSION` is stamped into every dataset; the reader refuses mismatches.

**Datasets** (`chessai.dataset`). One row per played move (`chessai/dataset/schema.py`): packed board, castling, en passant, 50-move counter, repetition count, move index, both ratings, both clocks, time spent, time control, final result for the mover, `is_target`, split, game index, and Stockfish evals when joined. A game's rows are contiguous and never span shards.

```sh
# Your games. Time split: oldest 85% train, 5% val, newest 10% test.
.venv/bin/python -m chessai.dataset.build archive data/players/hydra33092 --target hydra33092 --out data/datasets/hydra --legal
# Join Stockfish analysis (rerun as the PC analysis progresses; rows are verified against FEN + move).
.venv/bin/python -m chessai.dataset.evals data/datasets/hydra data/analysis/hydra33092/positions.jsonl
# The rating dataset: read-only SQLite access, hash split 90/5/5, Hydra33092's games excluded so they never leak into pretraining.
.venv/bin/python -m chessai.dataset.build bulk data/games.sqlite3 --exclude hydra33092 --out data/datasets/bulk
```

```python
from chessai.dataset.reader import PositionDataset
ds = PositionDataset('data/datasets/hydra')
idx = ds.select('train', target=True)                 # Hydra33092's own moves
for b in ds.torch_batches(idx, 1024, shuffle=True, history=2):
    b['planes'], b['features'], b['move'], b['legal']  # (B,41,8,8), (B,16), (B,), (B,1858)
```

`history=k` adds the previous k positions (re-oriented to the current mover). `where=` filters on any row column (e.g. `lambda r: r['time_class'] == 0` for bullet). Outputs live under `data/datasets/` and are not committed.

## Models (`chessai.models`)

The headline metric is **move match**: how often the model plays exactly the move Hydra33092 played, measured on the newest 10% of games, which are never trained on.

```sh
.venv/bin/python -m chessai.models.baselines                     # most-common-move and Stockfish-best baselines
.venv/bin/python -m chessai.models.train --data data/datasets/bulk --in-memory --out runs/base \
    --steps 60000 --eval data/datasets/hydra                      # general human model
.venv/bin/python -m chessai.models.train --data data/datasets/hydra --target-only \
    --init runs/base/best.pt --out runs/hydra --steps 20000 --lr 3e-4   # fine-tune on your moves
.venv/bin/python -m chessai.models.train --evaluate runs/hydra/best.pt --data data/datasets/hydra --split test
.venv/bin/python -m chessai.engine.play --white human --black model:path=runs/hydra/best.pt,temperature=1
```

`PolicyNet` (`chessai/models/net.py`) is a residual conv net (6×64 by default). Ratings, clocks and time control are broadcast in as extra input planes. It has three heads: move (1,858 logits), result (loss/draw/win) and log seconds spent. `chessai/models/inputs.py` builds live-game inputs through the same code as the dataset; a test checks they match the training rows exactly. `ModelPlayer.top_moves(game)` lists the predicted moves with their probabilities. Training runs on CUDA, MPS or CPU, chosen automatically (`--device` to override). Checkpoints and logs go to `runs/`, which is not committed.
