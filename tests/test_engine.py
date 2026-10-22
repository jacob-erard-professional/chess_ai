import glob
import json
import random
import shutil
import time
import unittest

import chess

from chessai.engine import (AlphaBetaPlayer, Game, GameOver, IllegalMove, PlayerInfo,
                            RandomPlayer, StockfishPlayer, TimeControl, play_game, tcn)
from chessai.engine.game import format_clock, parse_clocks
from chessai.engine.search import MATE, Searcher, evaluate


def played(moves, **kw):
    game = Game(**kw)
    for m in moves:
        game.play(m)
    return game


class RulesTest(unittest.TestCase):
    def test_castling_both_sides_and_lost_rights(self):
        g = played(['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5', 'O-O'])
        self.assertEqual(g.board.piece_at(chess.G1), chess.Piece(chess.KING, chess.WHITE))
        g = Game(fen='r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
        g.play('Ke2')
        g.play('O-O-O')
        self.assertEqual(g.board.piece_at(chess.D8), chess.Piece(chess.ROOK, chess.BLACK))
        g.play('Ke1')
        g.play('Kb8')
        with self.assertRaises(IllegalMove):
            g.play('O-O')  # the king moved, so White's rights are gone

    def test_cannot_castle_through_check(self):
        g = Game(fen='4k3/8/8/8/8/8/5r2/4K2R w K - 0 1')
        with self.assertRaises(IllegalMove):
            g.play('O-O')

    def test_en_passant(self):
        g = played(['e4', 'a6', 'e5', 'd5', 'exd6'])
        self.assertIsNone(g.board.piece_at(chess.D5))

    def test_en_passant_expires(self):
        g = played(['e4', 'a6', 'e5', 'd5', 'h3', 'h6'])
        with self.assertRaises(IllegalMove):
            g.play('exd6')

    def test_promotion_and_underpromotion(self):
        g = Game(fen='8/P6k/8/8/8/8/8/K7 w - - 0 1')
        g.play('a8=N')
        self.assertEqual(g.board.piece_type_at(chess.A8), chess.KNIGHT)
        g = Game(fen='8/P6k/8/8/8/8/8/K7 w - - 0 1')
        g.play('a7a8q')
        self.assertEqual(g.board.piece_type_at(chess.A8), chess.QUEEN)

    def test_checkmate(self):
        g = played(['f3', 'e5', 'g4', 'Qh4#'])
        self.assertEqual((g.result, g.reason, g.player_result(chess.WHITE)), ('0-1', 'checkmated', 'checkmated'))
        self.assertEqual(g.termination(), 'Black won by checkmate')
        with self.assertRaises(GameOver):
            g.play('a3')

    def test_stalemate(self):
        g = Game(fen='k7/8/1Q6/8/8/8/8/7K w - - 0 1')
        g.play('Qc7')
        self.assertEqual((g.result, g.reason), ('1/2-1/2', 'stalemate'))

    def test_threefold_is_automatic_like_chesscom(self):
        g = played(['Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3', 'Nf6', 'Ng1', 'Ng8'])
        self.assertEqual(g.reason, 'repetition')

    def test_threefold_must_be_claimed_under_fide(self):
        g = played(['Nf3', 'Nf6', 'Ng1', 'Ng8', 'Nf3', 'Nf6', 'Ng1', 'Ng8'], auto_draw=False)
        self.assertFalse(g.is_over)
        self.assertTrue(g.claim_draw())
        self.assertEqual(g.reason, 'repetition')

    def test_fifty_move_rule(self):
        g = Game(fen='4k3/8/8/8/8/8/8/R3K3 w - - 99 80')
        g.play('Ra2')
        self.assertEqual(g.reason, '50move')

    def test_chesscom_insufficient_material(self):
        for fen, move in [('8/b7/2k5/2N5/1pK5/8/8/8 w - - 0 73', 'Kxb4'),        # KN v KB
                          ('8/8/8/3k4/8/8/2n5/R1N1K3 b - - 0 60', 'Nxa1'),      # KN v KN
                          ('8/8/8/3k4/8/2b5/8/R2B3K b - - 0 60', 'Bxa1'),       # KB v KB, opposite colors
                          ('8/8/8/3k4/8/8/4p3/2N1KN2 w - - 0 60', 'Kxe2')]:     # KNN v K
            g = Game(fen=fen)
            g.play(move)
            self.assertEqual((g.result, g.reason), ('1/2-1/2', 'insufficient'), fen)
            g = Game(fen=fen, auto_draw=False)  # FIDE: a helpmate exists, so play on
            g.play(move)
            self.assertFalse(g.is_over, fen)
        g = Game(fen='8/8/8/3k4/8/8/4p3/2NBK3 w - - 0 60')  # KBN v K can be forced: play on
        g.play('Kxe2')
        self.assertFalse(g.is_over)

    @unittest.skipUnless(glob.glob('data/players/hydra33092/[0-9]*.json'), 'needs local archive')
    def test_archived_games_end_exactly_when_chesscom_ended_them(self):
        insufficient = 0
        for path in sorted(glob.glob('data/players/hydra33092/[0-9]*.json')):
            with open(path) as f:
                games = json.load(f)['games']
            for data in games:
                if data['rules'] != 'chess' or '/' in data['time_control']:
                    continue
                g = Game(fen=data['initial_setup'])
                moves = tcn.decode(data['tcn'])
                for i, move in enumerate(moves):
                    self.assertFalse(g.is_over, f"{data['url']} ended early at ply {i}: {g.reason}")
                    g.play(tcn.legalize(g.board, move))
                if data['white']['result'] == 'insufficient':
                    self.assertEqual(g.reason, 'insufficient', data['url'])
                    insufficient += 1
        self.assertGreater(insufficient, 40)

    def test_insufficient_material(self):
        g = Game(fen='4k3/8/8/8/8/8/3r4/3BK3 w - - 0 1')
        g.play('Kxd2')
        self.assertEqual(g.reason, 'insufficient')


class ClockTest(unittest.TestCase):
    def test_increment_and_move_times(self):
        g = Game(time_control='60+1')
        g.play('e4', elapsed=2.5)
        g.play('e5', elapsed=1.0)
        self.assertEqual(g.clocks[chess.WHITE], 58.5)
        self.assertEqual(g.move_times(), [2.5, 1.0])

    def test_flag(self):
        g = Game(time_control='10+0')
        self.assertIsNone(g.play('e4', elapsed=11))
        self.assertEqual((g.result, g.reason), ('0-1', 'timeout'))
        self.assertEqual(g.termination(), 'Black won on time')

    def test_timeout_vs_insufficient_material(self):
        g = Game(fen='4k3/8/8/8/8/8/8/Q3K3 w - - 0 1', time_control='10+0')
        g.play('Qa2', elapsed=11)
        self.assertEqual((g.result, g.reason), ('1/2-1/2', 'timevsinsufficient'))

    def test_timeout_vs_lone_minor_is_a_draw_even_if_helpmate_exists(self):
        # chess.com game 73455581879: Black flags, White has K+B (Black has a rook and knight).
        g = Game(fen='8/7p/4p2k/6p1/1B1K4/8/r7/3n4 b - - 3 43', time_control='10+0')
        g.flag(chess.BLACK)
        self.assertEqual((g.result, g.reason), ('1/2-1/2', 'timevsinsufficient'))
        g = Game(fen='8/7p/4p2k/6p1/1B1K4/8/r7/3n4 b - - 3 43', time_control='10+0')
        g.flag(chess.WHITE)  # Black has R+N+pawns: a win
        self.assertEqual((g.result, g.reason), ('0-1', 'timeout'))

    @unittest.skipUnless(glob.glob('data/players/hydra33092/[0-9]*.json'), 'needs local archive')
    def test_archived_timeouts_recompute_through_flag(self):
        checked = 0
        for path in sorted(glob.glob('data/players/hydra33092/[0-9]*.json')):
            with open(path) as f:
                games = json.load(f)['games']
            for data in games:
                codes = (data['white']['result'], data['black']['result'])
                if data['rules'] != 'chess' or '/' in data['time_control'] \
                        or not {'timeout', 'timevsinsufficient'} & set(codes):
                    continue
                g = Game.from_chesscom(data)
                g.result = g.reason = g.winner = None
                g.flag(g.board.turn)  # the side to move is the one whose clock ran out
                self.assertEqual((g.player_result(chess.WHITE), g.player_result(chess.BLACK)), codes, data['url'])
                checked += 1
        self.assertGreater(checked, 5000)

    def test_time_control_parsing(self):
        self.assertEqual(TimeControl.parse('180+2'), TimeControl(180, 2))
        self.assertEqual(TimeControl.parse('60').time_class, 'bullet')
        self.assertEqual(TimeControl.parse('180').time_class, 'blitz')
        self.assertEqual(TimeControl.parse('600').time_class, 'rapid')
        self.assertIsNone(TimeControl.parse('-'))
        self.assertEqual(format_clock(10.0), '0:00:10')
        self.assertEqual(format_clock(65.94), '0:01:05.9')


class RecordTest(unittest.TestCase):
    def test_chesscom_round_trip(self):
        g = Game(PlayerInfo('alice', 1200), PlayerInfo('bob', 1250), time_control='60+1')
        for move, t in [('e4', 1.2), ('c5', 0.8), ('Nf3', 3.1), ('d6', 0.4), ('Bb5+', 2.0)]:
            g.play(move, t)
        g.resign(chess.BLACK)
        record = g.to_chesscom()
        self.assertEqual((record['white']['result'], record['black']['result']), ('win', 'resigned'))
        self.assertIn('[Termination "alice won by resignation"]', record['pgn'])
        self.assertEqual(len(parse_clocks(record['pgn'])), 5)
        back = Game.from_chesscom(record)
        self.assertEqual(back.moves, g.moves)
        self.assertEqual(back.move_times(), [round(t, 3) for t in g.move_times()])
        self.assertEqual(back.to_chesscom()['pgn'], record['pgn'])

    def test_tcn_round_trip_with_promotions(self):
        g = Game(fen='8/P6k/8/8/8/8/1p5K/8 w - - 0 1')
        for m in ['a8=R', 'b1=N', 'Rb8', 'Nd2']:
            g.play(m)
        self.assertEqual(tcn.decode(tcn.encode(g.moves)), g.moves)

    def test_tcn_bare_queen_promotion_is_recovered(self):
        board = chess.Board('5r2/6P1/2p1p3/3p2r1/pk1P4/2nKP3/P6P/8 w - - 0 36')
        move = tcn.legalize(board, tcn.decode(tcn.ALPHABET[chess.G7] + tcn.ALPHABET[chess.F8])[0])
        self.assertEqual(move.uci(), 'g7f8q')

    @unittest.skipUnless(glob.glob('data/players/hydra33092/2025-02.json'), 'needs local archive')
    def test_real_games_reproduce_exactly(self):
        with open('data/players/hydra33092/2025-02.json') as f:
            games = json.load(f)['games']
        for data in games:
            if data['rules'] != 'chess':
                continue
            out = Game.from_chesscom(data).to_chesscom()
            self.assertEqual(out['tcn'], data['tcn'])
            self.assertEqual(out['pgn'].split('\n\n')[1], data['pgn'].split('\n\n')[1].rstrip('\n') + '\n')
            # chess.com sometimes drops the move counters from `fen`
            self.assertEqual(out['fen'].split()[:4], data['fen'].split()[:4])
            for side in ('white', 'black'):
                self.assertEqual(out[side]['result'], data[side]['result'])


class PlayersTest(unittest.TestCase):
    def test_bot_game_finishes_and_records(self):
        g = play_game(AlphaBetaPlayer(depth=1), RandomPlayer(seed=3), '-')
        self.assertTrue(g.is_over)
        self.assertEqual(Game.from_chesscom(g.to_chesscom()).moves, g.moves)

    def test_search_finds_mate_in_one(self):
        move, score = Searcher(2).search(chess.Board('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1'))
        self.assertEqual((move.uci(), score), ('a1a8', MATE - 1))
        board = chess.Board('r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4')
        self.assertEqual(Searcher(2).search(board)[0].uci(), 'h5f7')

    def test_search_finds_mate_in_two(self):
        # Rd8+ Rxd8 Rxd8#: only the rook with a backup on the d-file mates
        board = chess.Board('r5k1/5ppp/8/8/8/8/3R1PPP/3R2K1 w - - 0 1')
        move, score = Searcher(3).search(board)
        self.assertEqual(score, MATE - 3)
        self.assertEqual(move.uci(), 'd2d8')
        move, score = Searcher(3).search(board.mirror())
        self.assertEqual(score, MATE - 3)

    def test_quiescence_sees_mate_at_the_horizon(self):
        # Rxa7?? Qxg2#: depth 1 must see the reply mate inside quiescence.
        board = chess.Board('6rk/r7/6q1/8/8/8/5PPP/R5K1 w - - 0 1')
        move, score = Searcher(1).search(board)
        self.assertNotEqual(move.uci(), 'a1a7')
        self.assertGreater(score, -MATE + 100)

    def test_quiescence_scores_stalemate_as_draw(self):
        # Black to move is stalemated: quiescence must not return the material eval.
        self.assertEqual(Searcher(1).quiesce(chess.Board('k7/2Q5/1K6/8/8/8/8/8 b - - 0 1'), -MATE, MATE, 0), 0)

    def test_time_budget_deepens_past_default_depth(self):
        player = AlphaBetaPlayer(time=1.5)
        started = time.monotonic()
        player.choose_move(Game(time_control='-'))
        self.assertGreater(time.monotonic() - started, 1.0)
        searcher = Searcher(AlphaBetaPlayer.TIMED_MAX_DEPTH, 1.5)
        searcher.search(chess.Board())
        self.assertGreater(searcher.depth_reached, AlphaBetaPlayer.DEFAULT_DEPTH)
        self.assertEqual(AlphaBetaPlayer(depth=2).name, 'alphabeta-d2')

    @unittest.skipUnless(shutil.which('stockfish'), 'needs stockfish')
    def test_stockfish_elo_is_validated(self):
        for elo in (1000, 4000):
            with self.assertRaisesRegex(ValueError, '1320-3190'):
                StockfishPlayer(elo=elo)
        for elo in (1320, 3190):
            player = StockfishPlayer(elo=elo, depth=1)
            self.assertIn(player.choose_move(Game()), Game().legal_moves())
            player.close()

    def test_search_without_legal_moves(self):
        self.assertEqual(Searcher(2).search(chess.Board('7k/5QQ1/8/8/8/8/8/K7 b - - 0 1')), (None, -MATE))
        self.assertEqual(Searcher(2).search(chess.Board('k7/2Q5/1K6/8/8/8/8/8 b - - 0 1')), (None, 0))

    def test_converts_basic_endgames_against_random_defence(self):
        for fen in ('7k/8/8/8/8/8/8/K5Q1 w - - 0 1', 'k7/8/8/8/8/8/8/3QK3 w - - 0 1',
                    '7k/8/8/8/8/8/8/K5R1 w - - 0 1', '8/8/3k4/8/8/8/8/R3K3 w - - 0 1'):
            for seed in range(3):
                g = play_game(AlphaBetaPlayer(depth=4), RandomPlayer(seed), '-', fen=fen, clock='free')
                self.assertEqual((g.result, g.reason), ('1-0', 'checkmated'), (fen, seed))

    def test_eval_is_color_symmetric(self):
        rng = random.Random(0)
        for _ in range(200):
            board = chess.Board()
            for _ in range(rng.randrange(120)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            self.assertEqual(evaluate(board), evaluate(board.mirror()), board.fen())

    def test_search_wins_material(self):
        board = chess.Board('4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1')  # the black king cannot recapture
        self.assertEqual(Searcher(3).search(board)[0].uci(), 'd1d5')


if __name__ == '__main__':
    unittest.main()
