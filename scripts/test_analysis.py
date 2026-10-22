import unittest
import chess
from analyze_games import exchange_gain, sacrifice_evidence, score_data, init_worker, analyze_position
import analyze_games


class LegalSacrifices(unittest.TestCase):
    def test_pinned_attacker_cannot_take_offered_knight(self):
        board = chess.Board('4k3/4p3/8/5N2/8/8/8/K3R3 w - - 0 1')
        move = chess.Move.from_uci('f5d6')
        self.assertIn(move, board.legal_moves)
        after = board.copy()
        after.push(move)
        self.assertTrue(after.is_attacked_by(chess.BLACK, chess.D6))
        self.assertNotIn(chess.Move.from_uci('e7d6'), after.legal_moves)
        self.assertIsNone(sacrifice_evidence(board, move))

    def test_equal_exchange_is_not_a_sacrifice(self):
        board = chess.Board('4k3/8/4p3/3p4/2P5/8/8/4K3 w - - 0 1')
        self.assertIsNone(sacrifice_evidence(board, chess.Move.from_uci('c4d5')))

    def test_queen_offer_has_material_evidence(self):
        board = chess.Board('4k3/2p5/8/8/8/8/7Q/4K3 w - - 0 1')
        move = chess.Move.from_uci('h2d6')
        self.assertIn(move, board.legal_moves)
        evidence = sacrifice_evidence(board, move)
        self.assertEqual(evidence['net_material_offered_cp'], 900)

    def test_sound_queen_sacrifice_for_mate_is_a_candidate(self):
        import shutil
        init_worker(shutil.which('stockfish'), {'depth': 10, 'seconds': 0.5,
            'confirm_depth': 12, 'confirm_seconds': 0.5})
        board = chess.Board('5r1k/6pp/4Q2N/8/8/8/6PP/6K1 w - - 0 1')
        _, result = analyze_position(('queen-sacrifice', board.fen(), [], 'e6g8'))
        self.assertIsNotNone(result['brilliant'])
        self.assertEqual(result['brilliant']['label'], 'brilliant_candidate')
        self.assertGreater(result['brilliant']['evidence']['net_material_offered_cp'], 0)
        self.assertGreater(result['brilliant']['confirmed_played']['score']['mate'], 0)

    def test_mate_order_and_black_perspective(self):
        import chess.engine
        white_wins = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
        black_wins = chess.engine.PovScore(chess.engine.Mate(-3), chess.WHITE)
        self.assertLess(score_data(white_wins, chess.BLACK)['ordering_cp'], -90000)
        self.assertGreater(score_data(black_wins, chess.BLACK)['ordering_cp'], 90000)

    def test_engine_covers_every_legal_move_and_keeps_perspective(self):
        import shutil
        init_worker(shutil.which('stockfish'), {'depth': 6, 'seconds': 0.2,
            'confirm_depth': 8, 'confirm_seconds': 0.2})
        try:
            board = chess.Board('7k/8/5KQ1/8/8/8/8/8 w - - 0 1')
            move = chess.Move.from_uci('g6g7')
            _, result = analyze_position(('test', board.fen(), [], move.uci()))
            self.assertEqual(result['legal_move_count'], result['evaluated_move_count'])
            self.assertEqual(result['evaluated_move_count'], len(list(board.legal_moves)))
            self.assertEqual(result['played']['score']['mate'], 1)
            scores = [line['score']['ordering_cp'] for line in result['legal_moves_ranked']]
            self.assertEqual(result['worst_moves'][0]['score']['ordering_cp'], min(scores))
            self.assertEqual(result['best']['score']['ordering_cp'], max(scores))
        finally:
            analyze_games.close_engine()


if __name__ == '__main__':
    unittest.main()
