import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from chessai import encoding
from chessai.dataset.extract import extract
from chessai.dataset.reader import PositionDataset, _flip, features
from chessai.dataset.schema import pack_pieces, unpack_pieces
from chessai.engine import Game, PlayerInfo


def random_game(seed, plies=60):
    rng = random.Random(seed)
    g = Game(PlayerInfo('me', 1400), PlayerInfo('them', 1500), time_control='60+1')
    while not g.is_over and len(g.moves) < plies:
        g.play(rng.choice(g.legal_moves()), elapsed=rng.random())
    if not g.is_over:
        g.resign(chess.BLACK)
    return g


class EncodingTest(unittest.TestCase):
    def test_vocabulary(self):
        self.assertEqual(encoding.NUM_MOVES, 1858)
        self.assertEqual(len(set(encoding.MOVES)), 1858)

    def test_every_legal_move_round_trips(self):
        for seed in range(30):
            board = chess.Board()
            rng = random.Random(seed)
            for _ in range(120):
                moves = list(board.legal_moves)
                if not moves:
                    break
                for m in moves:
                    self.assertEqual(encoding.index_to_move(encoding.move_to_index(m, board.turn), board), m)
                board.push(rng.choice(moves))

    def test_perspective_symmetry(self):
        # The same position with colors swapped encodes identically.
        board = chess.Board('r3k2r/pp3ppp/2n5/3pP3/8/8/PPP2PPP/R3K2R w KQkq d6 0 1')
        self.assertEqual(encoding.encode_board(board)[1:], encoding.encode_board(board.mirror())[1:])
        np.testing.assert_array_equal(encoding.encode_board(board)[0], encoding.encode_board(board.mirror())[0])
        self.assertEqual(encoding.encode_board(board)[2], 3)  # en passant on the d-file

    def test_decode_board_is_inverse(self):
        board = chess.Board('r3k2r/pp3ppp/2n5/3pP3/8/8/PPP2PPP/R3K2R w Kq d6 0 1')
        self.assertEqual(encoding.decode_board(*encoding.encode_board(board)).fen().split()[:4],
                         board.fen().split()[:4])

    def test_planes(self):
        pieces, castling, ep = encoding.encode_board(chess.Board())
        planes = encoding.board_planes(pieces, castling, ep)
        self.assertEqual(planes.shape, (1, 17, 8, 8))
        self.assertEqual(planes[0, :12].sum(), 32)
        self.assertEqual(planes[0, 0, 1].sum(), 8)  # own pawns on rank 2
        self.assertTrue((planes[0, 12:16] == 1).all())

    def test_pack_pieces(self):
        pieces = np.random.default_rng(0).integers(0, 13, (5, 64), dtype=np.uint8)
        np.testing.assert_array_equal(unpack_pieces(pack_pieces(pieces)), pieces)

    def test_flip_is_the_other_sides_view(self):
        board = chess.Board('r3k2r/pp3ppp/2n5/3pP3/8/8/PPP2PPP/R3K2R w KQkq - 0 1')
        white_view = encoding.encode_board(board)[0]
        board.turn = chess.BLACK
        np.testing.assert_array_equal(_flip(encoding.encode_board(board)[0][None])[0], white_view)


class ExtractTest(unittest.TestCase):
    def test_rows_match_game(self):
        g = random_game(1)
        rows, masks = extract(g.to_chesscom(), 7, 0, targets={'me'}, legal=True)
        self.assertEqual(len(rows), len(g.moves))
        replay = chess.Board()
        legal = np.unpackbits(masks, axis=1, count=encoding.NUM_MOVES).astype(bool)
        for i, move in enumerate(g.moves):
            self.assertEqual(encoding.index_to_move(rows['move'][i], replay), move)
            self.assertTrue(legal[i, rows['move'][i]])
            self.assertEqual(legal[i].sum(), replay.legal_moves.count())
            self.assertEqual(rows['is_target'][i], replay.turn == chess.WHITE)
            self.assertEqual(rows['elo'][i], 1400 if replay.turn else 1500)
            replay.push(move)
        np.testing.assert_allclose(rows['move_time'], g.move_times(), atol=0.101)  # PGN clocks have 0.1 s resolution
        self.assertEqual(rows['clock'][0], 60)
        self.assertTrue((rows['game'] == 7).all())
        self.assertEqual(set(rows['result'][rows['is_target'] == 1]), {1})  # black resigned

    def test_repetition_counter(self):
        g = Game(auto_draw=False)
        for m in ['Nf3', 'Nf6', 'Ng1', 'Ng8'] * 2 + ['e4']:
            g.play(m)
        g.resign(chess.WHITE)
        rows, _ = extract(g.to_chesscom(), 0, 0)
        self.assertEqual(list(rows['repetition'][[0, 4, 8]]), [1, 2, 3])


class BuildAndReadTest(unittest.TestCase):
    def test_build_archive_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            games = [random_game(seed).to_chesscom() for seed in range(120)]
            for i, g in enumerate(games):
                g['end_time'] = 1_700_000_000 + i
            (tmp / 'src').mkdir()
            (tmp / 'src' / '2024-01.json').write_text(json.dumps({'games': games}))
            subprocess.run([sys.executable, '-m', 'chessai.dataset.build', 'archive', str(tmp / 'src'),
                            '--target', 'ME', '--out', str(tmp / 'ds'), '--legal', '--workers', '2',
                            '--shard-rows', '500'], check=True, capture_output=True)
            ds = PositionDataset(tmp / 'ds')
            self.assertGreater(len(ds.shards), 1)
            self.assertEqual(len(ds), sum(len(g['tcn']) // 2 for g in games))
            self.assertEqual(len(ds.games), 120)
            # time split: newest games are test
            splits = [g['split'] for g in ds.games]
            self.assertEqual(splits[-12:], [2] * 12)
            self.assertEqual(splits[0], 0)
            target = ds.select('train', target=True)
            self.assertTrue((ds.rows(target)['is_target'] == 1).all())
            batch = ds.batch(target[:64], history=2)
            self.assertEqual(batch['planes'].shape, (64, 17 + 24, 8, 8))
            self.assertTrue(batch['legal'][np.arange(64), batch['move']].all())
            self.assertEqual(features(ds.rows(target[:3])).shape[1], 16)
            # history: the position one ply back, re-oriented, equals the current
            # board with the last move undone
            i = int(target[5])
            row = ds.rows([i])[0]
            g = ds.games[row['game']]
            board = Game.from_chesscom(next(x for x in games if x['uuid'] == g['uuid'])).board
            while len(board.move_stack) > row['ply'] - 1:
                board.pop()
            if row['ply'] >= 1:
                prev = encoding.board_planes(*encoding.encode_board(board))[0, :12]
                cur_view = ds.planes([i], history=1)[0, 17:29]
                flipped = prev.reshape(12, 8, 8)[:, ::-1]
                expected = np.concatenate([flipped[6:], flipped[:6]])
                np.testing.assert_array_equal(cur_view, expected)


if __name__ == '__main__':
    unittest.main()
