import unittest
from collect_games import connect, profile_game_count, profile_gate, counts, report, target_priority, enqueue_job, prioritize_player, KINDS


class ProfileFilter(unittest.TestCase):
    def setUp(self):
        self.db = connect(':memory:')
        self.db.executemany('INSERT INTO jobs(path) VALUES (?)',
            [('player/example/games/archives',), ('player/example/games/2026/10',)])
        self.db.commit()

    def tearDown(self):
        self.db.close()

    def test_counts_rated_live_games_not_puzzles_or_daily(self):
        stats = {'chess_rapid': {'record': {'win': 400, 'loss': 300, 'draw': 10}},
                 'chess_blitz': {'record': {'win': 200, 'loss': 100, 'draw': 5}},
                 'chess_daily': {'record': {'win': 5000}}, 'tactics': {'highest': {'rating': 2000}}}
        self.assertEqual(profile_game_count(stats), 1015)
        self.assertEqual(profile_game_count({}), 0)

    def test_unknown_player_must_be_checked_before_archive_fetch(self):
        self.assertEqual(profile_gate(self.db, 'player/example/games/2026/10', 1000), 'pending')
        self.assertEqual(self.db.execute("SELECT count(*) FROM jobs WHERE path='player/example/stats'").fetchone()[0], 1)
        self.assertEqual(profile_gate(self.db, 'player/example/games/archives', 1000), 'pending')
        self.assertEqual(self.db.execute("SELECT count(*) FROM jobs WHERE path='player/example/stats'").fetchone()[0], 1)

    def test_small_profile_skips_all_queued_archives(self):
        self.db.execute('INSERT INTO profiles VALUES (?,?,?)', ('example', 999, 0))
        self.assertEqual(profile_gate(self.db, 'player/example/games/archives', 1000), 'excluded')
        self.assertEqual(self.db.execute('SELECT count(*) FROM jobs WHERE done=0').fetchone()[0], 0)

    def test_unverified_games_do_not_fill_filtered_quotas(self):
        self.db.execute('INSERT INTO games VALUES (?,?,?,?)', ('game1', 1200, 'rapid', '{}'))
        self.assertEqual(counts(self.db), {})
        self.db.execute('INSERT INTO game_sources VALUES (?,?)', ('game1', 'example'))
        self.db.execute('INSERT INTO game_sources VALUES (?,?)', ('game1', 'another'))
        self.assertEqual(counts(self.db), {(1200, 'rapid'): 1})

    def test_progress_uses_supplied_counts_without_recount(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            with patch('collect_games.counts', side_effect=AssertionError('Unexpected recount')):
                result = report(self.db, Path(directory), 25000, {(1200, 'rapid'): 42})
            self.assertEqual(result['total'], 42)
            self.assertEqual(result['profile_filter']['games_with_verified_source'], 42)

    def test_threshold_is_inclusive(self):
        self.db.execute('INSERT INTO profiles VALUES (?,?,?)', ('example', 1000, 0))
        self.assertEqual(profile_gate(self.db, 'player/example/games/2026/10', 1000), 'ready')
        self.assertEqual(profile_gate(self.db, 'player/example/stats', 1000), 'ready')

    def test_targets_unfilled_rating_and_time_control(self):
        current = {(b, k): 25000 for b in range(600, 3000, 100) for k in KINDS}
        current[2500, 'rapid'] = 0
        def stats(rating, kind='rapid'):
            return {'chess_' + kind: {'last': {'rating': rating}, 'record': {'win': 3000}}}
        self.assertGreater(target_priority(stats(2550), current, 25000), 0)
        self.assertEqual(target_priority(stats(1400), current, 25000), 0)
        self.assertEqual(target_priority(stats(2550, 'blitz'), current, 25000), 0)
        current[2500, 'rapid'] = 25000
        self.assertEqual(target_priority(stats(2550), current, 25000), 0)

    def test_higher_deficits_get_higher_priority(self):
        stats = {'chess_rapid': {'last': {'rating': 2550}, 'record': {'win': 3000}}}
        current = {(b, k): 25000 for b in range(600, 3000, 100) for k in KINDS}
        current[2500, 'rapid'] = 0
        high = target_priority(stats, current, 25000)
        current[2500, 'rapid'] = 24900
        self.assertGreater(high, target_priority(stats, current, 25000))

    def test_stats_check_is_not_starved_by_high_priority_archive(self):
        enqueue_job(self.db, 'player/example/games/archives', 1500)
        self.assertEqual(profile_gate(self.db, 'player/example/games/archives', 1000), 'pending')
        first = self.db.execute('SELECT path FROM jobs WHERE done=0 ORDER BY priority DESC LIMIT 1').fetchone()[0]
        self.assertEqual(first, 'player/example/stats')

    def test_deferring_history_preserves_jobs_and_other_players(self):
        enqueue_job(self.db, 'player/example2/games/archives', 500)
        prioritize_player(self.db, 'example', -100)
        self.assertEqual(self.db.execute("SELECT priority FROM jobs WHERE path='player/example2/games/archives'").fetchone()[0], 500)
        self.assertEqual(self.db.execute('SELECT count(*) FROM jobs WHERE done=0').fetchone()[0], 3)


if __name__ == '__main__':
    unittest.main()
