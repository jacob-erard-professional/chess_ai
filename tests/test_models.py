import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from chessai.dataset.reader import PositionDataset
from chessai.engine import Game, PlayerInfo, play_game, RandomPlayer
from chessai.models import net
from chessai.models.inputs import game_inputs
from chessai.models.player import ModelPlayer


def random_game(seed, plies=50):
    rng = random.Random(seed)
    g = Game(PlayerInfo('me', 1400), PlayerInfo('them', 1500), time_control='180+2', auto_draw=False)
    while not g.is_over and len(g.moves) < plies:
        g.play(rng.choice(g.legal_moves()), elapsed=rng.random() * 3)
    if not g.is_over:
        g.resign(chess.BLACK)
    return g


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls.tmp.name)
        cls.records = [random_game(s).to_chesscom() for s in range(30)]
        for i, r in enumerate(cls.records):
            r['end_time'] = 1_700_000_000 + i
        (tmp / 'src').mkdir()
        (tmp / 'src' / '2024-01.json').write_text(json.dumps({'games': cls.records}))
        subprocess.run([sys.executable, '-m', 'chessai.dataset.build', 'archive', str(tmp / 'src'), '--target', 'me',
                        '--out', str(tmp / 'ds'), '--legal', '--workers', '1'], check=True, capture_output=True)
        cls.ds = PositionDataset(tmp / 'ds')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_live_inputs_match_dataset(self):
        """A model sees identical inputs in play as in training."""
        rng = np.random.default_rng(0)
        for i in rng.choice(len(self.ds), 40, replace=False):
            row = self.ds.rows([i])[0]
            meta = self.ds.games[row['game']]
            game = Game.from_chesscom(next(r for r in self.records if r['uuid'] == meta['uuid']))
            # rewind the game (and its clocks) to the row's position
            moves, clocks = game.moves[:row['ply']], game.clock_history[:row['ply']]
            live = Game(game.players[True], game.players[False], time_control='180+2', auto_draw=False)
            for m in moves:
                live.board.push(m)
            for k, c in enumerate(clocks):
                live.clocks[chess.WHITE if k % 2 == 0 else chess.BLACK] = c
            planes, feats = game_inputs(live, history=2)
            b = self.ds.batch([i], history=2)
            np.testing.assert_array_equal(planes, b['planes'])
            np.testing.assert_allclose(feats, b['features'], rtol=1e-6)

    def test_train_eval_and_play(self):
        out = Path(self.tmp.name) / 'run'
        subprocess.run([sys.executable, '-m', 'chessai.models.train', '--data', str(self.ds.root), '--out', str(out),
                        '--steps', '20', '--batch', '64', '--blocks', '1', '--channels', '8', '--eval-every', '10',
                        '--log-every', '10', '--device', 'cpu'], check=True, capture_output=True)
        model, ckpt = net.load(out / 'best.pt')
        self.assertEqual(model.history, 2)
        player = ModelPlayer(str(out / 'best.pt'), temperature=1, device='cpu', seed=0)
        game = play_game(player, RandomPlayer(seed=1), '60+0', clock='free', max_plies=40)
        self.assertGreater(len(game.moves), 0)
        self.assertEqual(len(player.top_moves(Game(), 3)), 3)


if __name__ == '__main__':
    unittest.main()
