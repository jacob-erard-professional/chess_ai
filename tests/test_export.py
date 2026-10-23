import json
import random
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from chessai import encoding
from chessai.engine import Game, PlayerInfo, TimeControl
from chessai.models import net
from chessai.models.inputs import game_inputs
from chessai.export import web

HYDRA = Path('runs/hydra/best.pt')
BASE = Path('runs/base/best.pt')
# |logit| tolerance for the top-5 moves at each precision: int8 keeps 1% of the weights exact and the rest within
# 1/254 of their block's range, so logits move by a few hundredths at most
LOGIT_TOL = {'fp32': 1e-4, 'fp16': 0.1, 'int8': 0.5}


def random_game(seed, plies=60):
    rng = random.Random(seed)
    g = Game(PlayerInfo('me', 1400), PlayerInfo('them', 1500), time_control='60+1')
    while not g.is_over and len(g.moves) < plies:
        g.play(rng.choice(g.legal_moves()), elapsed=rng.random())
    if not g.is_over:
        g.resign(chess.BLACK)
    return g


def replay(fixture):
    """The live Game a fixture describes, built only from the fixture's own fields."""
    game = Game(PlayerInfo(fixture['white']['username'], fixture['white']['rating']),
                PlayerInfo(fixture['black']['username'], fixture['black']['rating']),
                time_control=fixture['time_control'], fen=fixture['fen'], auto_draw=False)
    for uci in fixture['moves']:
        game.board.push(chess.Move.from_uci(uci))
    if fixture['time_control']:
        game.clocks = {chess.WHITE: fixture['clocks']['white'], chess.BLACK: fixture['clocks']['black']}
    return game


class ExportTest(unittest.TestCase):
    def check_contract(self, out, names):
        manifest = json.loads((out / 'manifest.json').read_text())
        fixtures = json.loads((out / 'fixtures.json').read_text())
        self.assertEqual(manifest['encoding_version'], encoding.ENCODING_VERSION)
        self.assertEqual(manifest['moves']['uci'], [m.uci() for m in encoding.MOVES])
        self.assertEqual([f['name'] for f in manifest['features']['formulas']], manifest['features']['names'])
        self.assertEqual(set(manifest['models']), set(names))
        self.assertGreater(len(fixtures['fixtures']), 0)
        sessions = {name: web.OnnxModel(out / manifest['models'][name]['file']) for name in names}
        models = {name: net.load(manifest['models'][name]['checkpoint'])[0] for name in names}
        for fx in fixtures['fixtures']:
            game = replay(fx)
            board = game.board
            # the recorded planes, features and legal mask are what chessai builds from the same game
            planes, feats = game_inputs(game, 2)
            dense = np.zeros(planes[0].size, np.float32)
            dense[fx['planes']['indices']] = fx['planes']['values']
            np.testing.assert_array_equal(dense.reshape(fx['planes']['shape']), planes[0], err_msg=fx['id'])
            np.testing.assert_allclose(fx['features'], feats[0], rtol=1e-6, atol=1e-6, err_msg=fx['id'])
            self.assertEqual(fx['legal'], np.flatnonzero(encoding.legal_mask(board)).tolist(), fx['id'])
            self.assertEqual(fx['played_index'], encoding.move_to_index(chess.Move.from_uci(fx['played']), board.turn))
            self.assertEqual(fx['halfmove'], board.halfmove_clock)
            self.assertEqual(fx['repetition'], web.repetition_count(board))
            for name in names:
                model, precision = models[name], manifest['models'][name]['precision']
                expected = fx['expected'][name]
                torch_out = web.torch_outputs(model, planes, feats)
                onnx_out = sessions[name](planes, feats)
                # torch top-5 reproduces, onnxruntime reproduces its own record exactly, and the two agree
                self.assertEqual(expected['torch']['top5'],
                                 web.top_moves(torch_out['policy'][0], encoding.legal_mask(board), board))
                self.assertEqual(expected['onnx']['top5'],
                                 web.top_moves(onnx_out['policy'][0], encoding.legal_mask(board), board))
                for uci, logit in expected['torch']['top5']:
                    i = encoding.move_to_index(chess.Move.from_uci(uci), board.turn)
                    self.assertAlmostEqual(logit, float(onnx_out['policy'][0][i]), delta=LOGIT_TOL[precision],
                                           msg=f'{fx["id"]} {name} {uci}')
                np.testing.assert_allclose(onnx_out['value'][0], torch_out['value'][0], atol=LOGIT_TOL[precision])
                self.assertAlmostEqual(float(onnx_out['log_time'][0]), float(torch_out['log_time'][0]),
                                       delta=LOGIT_TOL[precision])
        return manifest, fixtures

    def test_tiny_model_round_trip(self):
        """A fresh small net exports, quantises, and its fixtures reproduce through chessai."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'src').mkdir()
            records = [random_game(s).to_chesscom() for s in range(12)]
            for i, r in enumerate(records):
                r['end_time'] = 1_700_000_000 + i
            (tmp / 'src' / '2024-01.json').write_text(json.dumps({'games': records}))
            torch.manual_seed(0)
            net.save(net.PolicyNet(blocks=1, channels=8), tmp / 'tiny.pt')
            manifest = web.export({'tiny': str(tmp / 'tiny.pt')}, tmp / 'out', tmp / 'src', 'me',
                                  fixtures=12, eval_positions=30)
            self.assertEqual(manifest['models']['tiny']['planes'], 41)
            self.assertEqual(manifest['models']['tiny']['features'], 16)
            _, fixtures = self.check_contract(tmp / 'out', ['tiny'])
            self.assertEqual(len(fixtures['fixtures']), 12)
            self.assertIn('first_ply', fixtures['tags'])

    @unittest.skipUnless(HYDRA.exists() and BASE.exists(), 'needs the trained checkpoints in runs/')
    def test_trained_checkpoints(self):
        """The real export: int8 stays within tolerance of torch and covers every fixture category."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            manifest = web.export({'jacob': str(HYDRA), 'human': str(BASE)}, out, 'data/players/hydra33092',
                                  'hydra33092', fixtures=50, eval_positions=100)
            manifest, fixtures = self.check_contract(out, ['jacob', 'human'])
            for name, info in manifest['models'].items():
                drift = info['drift_vs_torch'][info['precision']]
                self.assertGreaterEqual(drift['argmax_agreement_clear'], 0.99, name)
                self.assertLess(abs(drift['nll_drift']), 0.02, name)
            self.assertFalse([t for t in web.TAG_QUOTA if t not in fixtures['tags']])
            self.assertEqual(manifest['play']['clone']['time_class'], TimeControl.parse('60').time_class)
            total = sum(p.stat().st_size for p in out.iterdir())
            self.assertLess(total, 12e6, f'{total / 1e6:.1f} MB exported')


if __name__ == '__main__':
    unittest.main()
