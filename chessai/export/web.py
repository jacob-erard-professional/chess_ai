"""Export checkpoints for the browser: ONNX files, the input contract, and golden fixtures.

    python -m chessai.export.web --model jacob=runs/hydra/best.pt --model human=runs/base/best.pt --out web/

Writes `<name>.onnx` per model, `manifest.json` (everything a JavaScript encoder
needs to build the inputs without reading this package: plane layout, feature
formulas, move vocabulary, play settings) and `fixtures.json` (positions from
real games with the expected planes, features, legal mask and top moves from
torch and onnxruntime). The smallest precision whose moves still agree with
torch is kept: int8 dynamic quantisation, then fp16, then fp32.
"""
import argparse
import collections
import glob
import json
import shutil
import statistics
import warnings
from pathlib import Path

import chess
import numpy as np
import onnx
import onnxruntime as ort
import torch

from .. import encoding
from ..dataset.reader import FEATURES, NUM_FEATURES
from ..dataset.schema import TIME_CLASSES, UNKNOWN_CLASS
from ..engine import Game
from ..models import net
from ..models.inputs import game_inputs

CONTRACT_VERSION = 1
OPSET = 17
PRECISIONS = ('int8', 'fp16', 'fp32')  # tried in this order
ALL_PRECISIONS = PRECISIONS + ('int8-dynamic',)
TAGS = ('first_ply', 'one_earlier', 'white_to_move', 'black_to_move', 'castling_partial', 'no_castling',
        'en_passant', 'promotion', 'repetition', 'halfmove_high', 'other')
TAG_QUOTA = {'first_ply': 2, 'one_earlier': 2, 'en_passant': 5, 'promotion': 5, 'repetition': 5,
             'halfmove_high': 4, 'castling_partial': 6, 'no_castling': 4, 'black_to_move': 6, 'white_to_move': 6,
             'other': 5}
HALFMOVE_HIGH = 30


# --- ONNX ----------------------------------------------------------------------
class _Heads(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, planes, features):
        out = self.model(planes, features)
        return out['policy'], out['value'], out['log_time']


def export_fp32(model, path):
    planes = torch.zeros(1, net.BOARD_PLANES + net.HISTORY_PLANES * model.history, 8, 8)
    features = torch.zeros(1, model.config['num_features'])
    torch.onnx.export(_Heads(model).eval(), (planes, features), str(path), opset_version=OPSET, dynamo=False,
                      input_names=['planes', 'features'], output_names=['policy', 'value', 'log_time'],
                      do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(path)))


def _block(cols, block):
    """Largest divisor of cols that is at most block (the whole row when none is 8 or more)."""
    return max((b for b in range(1, block + 1) if cols % b == 0 and b >= 8), default=cols)


def quantize_weights(model, block=32, outliers=0.01):
    """int8 weights for every Conv and Gemm, dequantised to float32 at run time.

    The policy layer mixes a few huge weights with many small ones, clustered
    by input feature, so one scale per output row loses the small weights.
    Each weight is therefore stored as blocks of `block` values with their
    own scale (Gemm weights are blocked along the input axis, so the Gemm
    becomes transB=0) and the largest `outliers` fraction of entries are kept
    exactly in float32, scattered back after dequantisation. Everything is
    standard opset-13 ops (DequantizeLinear axis=0, Reshape, ScatterND) and
    the arithmetic stays float32, which is what keeps the policy within a hair
    of torch; dynamic activation quantisation does not (--precision int8-dynamic).
    """
    from onnx import numpy_helper
    layers = {node.input[1]: node for node in model.graph.node if node.op_type in ('Conv', 'Gemm')}
    keep, nodes = [], []
    for init in model.graph.initializer:
        node = layers.get(init.name)
        if node is None:
            keep.append(init)
            continue
        w = numpy_helper.to_array(init)
        if node.op_type == 'Gemm':
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            w = w.T if attrs.get('transB') else w  # (K, N): blocks run along the input axis
            del node.attribute[:]
            node.attribute.extend([onnx.helper.make_attribute(k, v) for k, v in attrs.items() if k != 'transB'])
        flat = w.reshape(w.shape[0], -1)
        k = int(round(flat.size * outliers))
        big = np.zeros(flat.shape, bool)
        if k:
            big.flat[np.argpartition(-np.abs(flat).ravel(), k - 1)[:k]] = True
        rest = np.where(big, 0, flat)
        b = _block(flat.shape[1], block)
        blocks = rest.reshape(-1, b)
        scale = np.abs(blocks).max(1) / 127
        scale = np.where(scale > 0, scale, 1).astype(np.float32)
        q = np.clip(np.rint(blocks / scale[:, None]), -127, 127).astype(np.int8)
        name = init.name
        keep += [numpy_helper.from_array(q, name + '_q'), numpy_helper.from_array(scale, name + '_scale'),
                 numpy_helper.from_array(np.zeros(len(scale), np.int8), name + '_zero'),
                 numpy_helper.from_array(np.array(w.shape, np.int64), name + '_shape')]
        nodes.append(onnx.helper.make_node('DequantizeLinear', [name + '_q', name + '_scale', name + '_zero'],
                                           [name + '_dq'], axis=0))
        dense = name + '_dq'
        if k:
            keep += [numpy_helper.from_array(np.array([-1], np.int64), name + '_flat_shape'),
                     numpy_helper.from_array(np.flatnonzero(big)[:, None].astype(np.int64), name + '_big_index'),
                     numpy_helper.from_array(flat[big].astype(np.float32), name + '_big')]
            nodes += [onnx.helper.make_node('Reshape', [dense, name + '_flat_shape'], [name + '_flat']),
                      onnx.helper.make_node('ScatterND', [name + '_flat', name + '_big_index', name + '_big'],
                                            [name + '_full'])]
            dense = name + '_full'
        nodes.append(onnx.helper.make_node('Reshape', [dense, name + '_shape'], [name]))
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    nodes.extend(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return model


def convert(fp32_path, path, precision):
    """Write `path` at `precision` from the fp32 export."""
    if precision == 'fp32':
        shutil.copyfile(fp32_path, path)
    elif precision == 'fp16':
        from onnxconverter_common import float16
        model = float16.convert_float_to_float16(onnx.load(str(fp32_path)), keep_io_types=True)
        onnx.save(model, str(path))
    elif precision == 'int8':
        model = quantize_weights(onnx.load(str(fp32_path)))
        onnx.checker.check_model(model)
        onnx.save(model, str(path))
    elif precision == 'int8-dynamic':
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32_path), str(path), weight_type=QuantType.QInt8)
    else:
        raise ValueError(f'unknown precision {precision!r}')


class OnnxModel:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])

    def __call__(self, planes, features):
        policy, value, log_time = self.session.run(None, {'planes': planes, 'features': features})
        return {'policy': policy, 'value': value, 'log_time': log_time}


@torch.no_grad()
def torch_outputs(model, planes, features):
    out = model(torch.from_numpy(planes), torch.from_numpy(features))
    return {k: v.numpy() for k, v in out.items()}


# --- positions from real games -----------------------------------------------------------
def load_archive(archive, player=None):
    """Replayed Games from monthly chess.com JSON, newest game first."""
    records = []
    for path in sorted(glob.glob(str(Path(archive) / '[0-9]*.json'))):
        records.extend(json.loads(Path(path).read_text())['games'])
    games = []
    for record in sorted(records, key=lambda r: -r.get('end_time', 0)):
        if record.get('rules', 'chess') != 'chess' or '/' in str(record.get('time_control', '')):
            continue
        if player and player.lower() not in (record['white']['username'].lower(), record['black']['username'].lower()):
            continue
        try:
            games.append(Game.from_chesscom(record))
        except (ValueError, KeyError):
            continue
    return games


def rewind(game, ply):
    """A live Game at `ply` moves into `game`, clocks as they stood before the move."""
    live = Game(game.players[chess.WHITE], game.players[chess.BLACK], time_control=game.time_control,
                fen=game.initial_fen, auto_draw=False)
    for move in game.moves[:ply]:
        live.board.push(move)
    for i, clock in enumerate(game.clock_history[:ply]):
        live.clocks[chess.WHITE if i % 2 == 0 else chess.BLACK] = clock
    return live


def repetition_count(board):
    key, seen, replay = board._transposition_key(), 1, board.copy()
    while replay.move_stack:
        replay.pop()
        seen += replay._transposition_key() == key
    return seen


def position_tags(board):
    tags = set()
    ply = len(board.move_stack)
    if ply == 0:
        tags.add('first_ply')
    if ply == 1:
        tags.add('one_earlier')
    tags.add('white_to_move' if board.turn == chess.WHITE else 'black_to_move')
    rights = sum(f(c) for c in chess.COLORS
                 for f in (board.has_kingside_castling_rights, board.has_queenside_castling_rights))
    if 0 < rights < 4:
        tags.add('castling_partial')
    if rights == 0 and ply > 0:
        tags.add('no_castling')
    if board.has_legal_en_passant():
        tags.add('en_passant')
    if any(m.promotion for m in board.legal_moves):
        tags.add('promotion')
    if repetition_count(board) > 1:
        tags.add('repetition')
    if board.halfmove_clock >= HALFMOVE_HIGH:
        tags.add('halfmove_high')
    if not tags - {'white_to_move', 'black_to_move'}:
        tags.add('other')
    return tags


def sample_positions(games, rng, player=None, max_games=800):
    """(game, ply, tags) covering every tag in TAG_QUOTA, from the newest games."""
    pools = collections.defaultdict(list)
    for game in games[:max_games]:
        board = chess.Board(game.initial_fen)
        for ply, move in enumerate(game.moves):
            mover = game.players[board.turn].username.lower()
            if player is None or mover == player.lower():
                for tag in position_tags(board):
                    pools[tag].append((game, ply))
            board.push(move)
    chosen, keys = [], set()
    for tag, quota in TAG_QUOTA.items():
        pool = pools[tag]
        for i in rng.permutation(len(pool))[:quota * 3]:
            game, ply = pool[i]
            if (game.uuid, ply) not in keys and sum(tag in t for *_, t in chosen) < quota:
                keys.add((game.uuid, ply))
                chosen.append((game, ply, sorted(position_tags(rewind(game, ply).board))))
    return chosen


def random_positions(games, rng, n, player=None, exclude=()):
    """n (game, ply) pairs spread over the games, for measuring quantisation drift."""
    candidates = []
    for game in games:
        board = chess.Board(game.initial_fen)
        for ply, move in enumerate(game.moves):
            if player is None or game.players[board.turn].username.lower() == player.lower():
                if (game.uuid, ply) not in exclude:
                    candidates.append((game, ply))
            board.push(move)
    return [candidates[i] for i in rng.permutation(len(candidates))[:n]]


# --- accuracy ----------------------------------------------------------------------
def _log_softmax(logits, legal):
    masked = np.where(legal, logits.astype(np.float64), -np.inf)
    masked -= masked.max()
    return masked - np.log(np.exp(masked).sum())


def compare(model, onnx_model, positions, clear_margin=0.02):
    """Drift of the ONNX policy from torch over (game, ply) positions.

    argmax_agreement counts every position; argmax_agreement_clear only those
    where torch's best move leads the runner-up by at least `clear_margin`
    probability, since near-ties flip on noise that is far below what
    sampling at temperature 1 notices.
    """
    agree, clear, clear_agree, nll_torch, nll_onnx, kl = 0, 0, 0, [], [], []
    max_logp, max_prob = 0.0, 0.0
    for game, ply in positions:
        live = rewind(game, ply)
        planes, features = game_inputs(live, model.history)
        legal = encoding.legal_mask(live.board)
        played = encoding.move_to_index(game.moves[ply], live.board.turn)
        a = _log_softmax(torch_outputs(model, planes, features)['policy'][0], legal)
        b = _log_softmax(onnx_model(planes, features)['policy'][0], legal)
        pa, pb = np.exp(a), np.exp(b)
        same = a.argmax() == b.argmax()
        agree += same
        if pa.max() - np.partition(pa, -2)[-2] >= clear_margin:
            clear += 1
            clear_agree += same
        nll_torch.append(-a[played])
        nll_onnx.append(-b[played])
        kl.append(float(np.sum(pa[legal] * (a[legal] - b[legal]))))
        max_logp = max(max_logp, float(np.abs(a[legal] - b[legal]).max()))
        max_prob = max(max_prob, float(np.abs(pa - pb).max()))
    return {'positions': len(positions), 'argmax_agreement': agree / len(positions),
            'clear_positions': clear, 'clear_margin': clear_margin,
            'argmax_agreement_clear': clear_agree / clear if clear else agree / len(positions),
            'nll_torch': statistics.fmean(nll_torch), 'nll_onnx': statistics.fmean(nll_onnx),
            'nll_drift': statistics.fmean(nll_onnx) - statistics.fmean(nll_torch),
            'kl_torch_onnx': statistics.fmean(kl), 'max_log_prob_diff': max_logp, 'max_prob_diff': max_prob}


def choose_precision(model, fp32_path, out_path, positions, precisions, min_agreement, max_nll_drift):
    """Convert to each precision in turn and keep the first that stays close enough to torch."""
    reports = {}
    for precision in precisions:
        try:
            convert(fp32_path, out_path, precision)
            onnx_model = OnnxModel(out_path)
            report = compare(model, onnx_model, positions)
        except Exception as e:  # a precision the runtime cannot load or run
            reports[precision] = {'error': f'{type(e).__name__}: {e}'}
            continue
        report['bytes'] = out_path.stat().st_size
        reports[precision] = report
        if report['argmax_agreement_clear'] >= min_agreement and abs(report['nll_drift']) <= max_nll_drift:
            return precision, reports
    raise SystemExit(f'no precision matches torch (clear argmax >= {min_agreement}, |nll drift| <= {max_nll_drift}): '
                     + json.dumps(reports, indent=1))


# --- fixtures ----------------------------------------------------------------------
def top_moves(logits, legal, board, k=5):
    order = np.argsort(-np.where(legal, logits, -np.inf))[:k]
    return [[encoding.index_to_move(int(i), board).uci(), float(logits[i])] for i in order]


def fixture(game, ply, tags, models):
    """One golden vector: the position, the exact inputs, and every model's outputs."""
    live = rewind(game, ply)
    board = live.board
    planes, features = game_inputs(live, max(m.history for m, _ in models.values()))
    legal = encoding.legal_mask(board)
    nonzero = np.flatnonzero(planes[0])
    tc = live.time_control
    expected = {}
    for name, (model, onnx_model) in models.items():
        p, f = game_inputs(live, model.history)
        t, o = torch_outputs(model, p, f), onnx_model(p, f)
        expected[name] = {'torch': {'top5': top_moves(t['policy'][0], legal, board),
                                    'value': t['value'][0].tolist(), 'log_time': float(t['log_time'][0])},
                          'onnx': {'top5': top_moves(o['policy'][0], legal, board),
                                   'value': o['value'][0].tolist(), 'log_time': float(o['log_time'][0])}}
    return {
        'id': f'{game.uuid}:{ply}', 'source': game.url, 'tags': tags,
        'fen': game.initial_fen, 'moves': [m.uci() for m in game.moves[:ply]],
        'white': {'username': game.players[chess.WHITE].username, 'rating': game.players[chess.WHITE].rating},
        'black': {'username': game.players[chess.BLACK].username, 'rating': game.players[chess.BLACK].rating},
        'time_control': str(tc) if tc else None, 'time_class': tc.time_class if tc else None,
        'clocks': {'white': live.clocks[chess.WHITE], 'black': live.clocks[chess.BLACK]},
        'position_fen': board.fen(), 'halfmove': board.halfmove_clock, 'repetition': repetition_count(board),
        'ply': ply, 'played': game.moves[ply].uci(), 'played_index': encoding.move_to_index(game.moves[ply], board.turn),
        'planes': {'shape': list(planes.shape[1:]), 'indices': nonzero.tolist(), 'values': planes[0].flat[nonzero].tolist()},
        'features': features[0].tolist(), 'legal': np.flatnonzero(legal).tolist(), 'expected': expected,
    }


# --- the contract ----------------------------------------------------------------------
def archive_stats(games, player):
    """What the player's games look like: time controls, ratings, pace."""
    moves_by_tc, latest, move_times = collections.Counter(), {}, []
    for game in games:
        tc = game.time_control
        if not tc:
            continue
        side = chess.WHITE if game.players[chess.WHITE].username.lower() == player.lower() else chess.BLACK
        own = sum(1 for i in range(len(game.moves)) if (i % 2 == 0) == side)
        moves_by_tc[(str(tc), tc.time_class)] += own
        if tc.time_class not in latest or (game.end_time or 0) > latest[tc.time_class][1]:
            latest[tc.time_class] = (game.players[side].rating, game.end_time or 0, str(tc))
        clocks = game.clock_history
        for i in range(2, len(clocks)):
            if (i % 2 == 0) == side:
                move_times.append(max(0.0, clocks[i - 2] - clocks[i] + tc.increment))
    total = sum(moves_by_tc.values())
    return {'games': len(games), 'moves': total,
            'moves_by_time_control': [{'time_control': tc, 'time_class': cls, 'moves': n, 'share': round(n / total, 4)}
                                      for (tc, cls), n in moves_by_tc.most_common()],
            'latest_rating': {cls: {'rating': r, 'end_time': t, 'time_control': tc} for cls, (r, t, tc) in latest.items()},
            'move_seconds_percentiles': {str(p): round(float(np.percentile(move_times, p)), 2) for p in (25, 50, 75, 90)}
            if move_times else {}}


def manifest(models, stats, player):
    bullet = stats['latest_rating'].get('bullet', {}).get('rating')
    return {
        'contract_version': CONTRACT_VERSION,
        'encoding_version': encoding.ENCODING_VERSION,
        'generated_by': 'chessai.export.web',
        'onnx': {
            'opset': OPSET,
            'inputs': {'planes': {'shape': [1, 'C', 8, 8], 'dtype': 'float32'},
                       'features': {'shape': [1, 'F'], 'dtype': 'float32'}},
            'outputs': {'policy': {'shape': [1, encoding.NUM_MOVES], 'meaning': 'move logits, index = moves.uci order; '
                                   'mask illegal entries to -inf then softmax (optionally logits / temperature first)'},
                        'value': {'shape': [1, 3], 'meaning': 'result logits for the side to move: [loss, draw, win]; softmax'},
                        'log_time': {'shape': [1], 'meaning': 'log1p(seconds the side to move spends on this move); '
                                     'seconds = expm1(log_time)'}},
            'batch': 1,
            'precisions': {'int8': 'int8 per-output-channel weights for every Conv and Gemm, expanded by '
                                   'DequantizeLinear (opset 13 form, axis=0) into float32 at run time, the largest '
                                   '1% of each layer\'s weights kept exactly via ScatterND; all arithmetic is '
                                   'float32. See models.*.drift_vs_torch for how close the policy stays',
                           'fp16': 'float16 weights and arithmetic, float32 inputs and outputs',
                           'fp32': 'the torch weights as they are'},
        },
        'models': {name: {'file': f'{name}.onnx', 'checkpoint': str(path), 'precision': precision,
                          'history': model.history, 'planes': net.BOARD_PLANES + net.HISTORY_PLANES * model.history,
                          'features': model.config['num_features'], 'bytes': report[precision]['bytes'],
                          'blocks': model.config['blocks'], 'channels': model.config['channels'],
                          'drift_vs_torch': report}
                   for name, (model, path, precision, report) in models.items()},
        'moves': {
            'count': encoding.NUM_MOVES,
            'uci': [m.uci() for m in encoding.MOVES],
            'description': 'Index order: for every from-square 0..63 (a1=0, b1=1, ..., h8=63) and every to-square '
                           '0..63 that a queen or a knight could reach on an empty board, the plain move from->to; '
                           'then for each from-square a7..h7 (48..55) and each capture-or-push to-square on rank 8 '
                           '(left, straight, right, when on the board) the underpromotions to n, b, r in that order. '
                           'Every string is in the side-to-move orientation: own pawns always move up the board.',
            'mirroring': 'When Black is to move, mirror both squares vertically before the lookup: '
                         'square ^ 56, i.e. rank r becomes rank 7-r and the file is unchanged '
                         '(a1<->a8, e2<->e7). When White is to move the squares are used as they are. '
                         'To turn an index back into a real move apply the same mirror to from and to.',
            'promotions': 'A queen promotion is the plain from->to entry (e.g. "e7e8" with no suffix); the vocabulary '
                          'never spells "q". Knight, bishop and rook promotions use the entries with the "n", "b", "r" '
                          'suffix, always from rank 7 to rank 8 in the side-to-move orientation. When decoding an '
                          'index into a move: if the entry has no suffix but the moving piece is a pawn landing on '
                          'rank 8 (rank 1 after mirroring), it is a queen promotion.',
            'legal_mask': 'legal[i] = 1 when moves.uci[i], after mirroring for Black and after adding the implied '
                          'queen promotion, is a legal move of the current position (castling is the king move '
                          'e1g1 / e1c1, en passant is the pawn move to the target square).',
        },
        'planes': {
            'shape': [net.BOARD_PLANES + net.HISTORY_PLANES * 2, 8, 8],
            'shape_note': 'C = 17 + 12 * history; both exported models use history 2, so C = 41.',
            'layout': 'planes[c][rank][file]; flat index c*64 + rank*8 + file; rank 0 = the side to move\'s back '
                      'rank (a1 = index 0 for White; for Black the board is mirrored so Black\'s back rank 8 is '
                      'rank 0). Files are never mirrored: file 0 is always the a-file.',
            'orientation': 'Everything is from the side to move\'s point of view: when Black is to move, every square '
                           'is mirrored vertically (square ^ 56) and the colours swap, so "own" pieces always '
                           'move up the board.',
            'channels': [
                {'index': 0, 'name': 'own pawn'}, {'index': 1, 'name': 'own knight'}, {'index': 2, 'name': 'own bishop'},
                {'index': 3, 'name': 'own rook'}, {'index': 4, 'name': 'own queen'}, {'index': 5, 'name': 'own king'},
                {'index': 6, 'name': 'opponent pawn'}, {'index': 7, 'name': 'opponent knight'},
                {'index': 8, 'name': 'opponent bishop'}, {'index': 9, 'name': 'opponent rook'},
                {'index': 10, 'name': 'opponent queen'}, {'index': 11, 'name': 'opponent king'},
                {'index': 12, 'name': 'own kingside castling right', 'constant': True},
                {'index': 13, 'name': 'own queenside castling right', 'constant': True},
                {'index': 14, 'name': 'opponent kingside castling right', 'constant': True},
                {'index': 15, 'name': 'opponent queenside castling right', 'constant': True},
                {'index': 16, 'name': 'en passant target', 'note': 'a single 1 at rank 5 (0-based, the rank a capturing '
                 'own pawn lands on) and the file of the en passant square, only when an en passant capture is '
                 'actually legal this move; otherwise the plane is all zero'},
                {'index': '17 + 12*(k-1) .. 28 + 12*(k-1)', 'name': 'position k plies ago, k = 1..history',
                 'note': 'the 12 piece planes (same order as channels 0-11) of the position k plies earlier, encoded '
                         'from the CURRENT side to move\'s point of view (so mirrored when Black is to move now, '
                         'regardless of who moved then); no castling or en passant planes. All zero when fewer than '
                         'k moves have been played from the start position of the game.'},
            ],
            'castling': 'A castling plane is all 64 cells = 1 when that side still has that right (the king and the '
                        'rook have not moved and the rook is still on its square), else all 0. "Own" is the side to move.',
            'piece_values': 'Every nonzero cell is exactly 1.0.',
        },
        'features': {
            'count': NUM_FEATURES,
            'names': list(FEATURES),
            'inputs': {
                'elo': 'the side to move\'s rating (integer), 0 when unknown',
                'opp_elo': 'the opponent\'s rating, 0 when unknown',
                'clock': 'seconds left on the side to move\'s clock before this move (after their last increment); '
                         'NaN when the game is untimed',
                'opp_clock': 'seconds left on the opponent\'s clock; NaN when untimed',
                'base': 'integer seconds of the base time control, min(int(base), 65535); 0 when untimed',
                'inc': 'integer seconds of increment, min(int(increment), 255); 0 when untimed',
                'time_class': '0 bullet, 1 blitz, 2 rapid, 255 unknown (untimed). Chess.com\'s rule: '
                              'expected = base + 40 * increment; bullet if expected < 180, blitz if < 600, else rapid',
                'halfmove': 'the half-move clock of the position (plies since the last capture or pawn move), min(n, 255)',
                'repetition': 'how many times the current position has occurred in this game including now (>= 1), '
                              'min(n, 255). Positions are equal when the piece placement, side to move, castling '
                              'rights and en passant square (only counted when an en passant capture is legal) are equal',
                'ply': 'number of moves played so far from the start position of the game (0 for the first move)',
            },
            'formulas': [
                {'index': 0, 'name': 'elo', 'formula': 'elo > 0 ? (elo - 1500) / 500 : 0'},
                {'index': 1, 'name': 'opp_elo', 'formula': 'opp_elo > 0 ? (opp_elo - 1500) / 500 : 0'},
                {'index': 2, 'name': 'elo_diff', 'formula': 'elo > 0 && opp_elo > 0 ? (elo - opp_elo) / 200 : 0'},
                {'index': 3, 'name': 'log_clock', 'formula': 'timed ? log1p(clock) / 5 : 0'},
                {'index': 4, 'name': 'log_opp_clock', 'formula': 'timed ? log1p(opp_clock) / 5 : 0'},
                {'index': 5, 'name': 'clock_frac', 'formula': 'timed ? clock / max(base, 1) : 0'},
                {'index': 6, 'name': 'opp_clock_frac', 'formula': 'timed ? opp_clock / max(base, 1) : 0'},
                {'index': 7, 'name': 'log_base', 'formula': 'log1p(base) / 7'},
                {'index': 8, 'name': 'inc', 'formula': 'inc / 10'},
                {'index': 9, 'name': 'bullet', 'formula': 'time_class == 0 ? 1 : 0'},
                {'index': 10, 'name': 'blitz', 'formula': 'time_class == 1 ? 1 : 0'},
                {'index': 11, 'name': 'rapid', 'formula': 'time_class == 2 ? 1 : 0'},
                {'index': 12, 'name': 'halfmove', 'formula': 'halfmove / 100'},
                {'index': 13, 'name': 'repetition', 'formula': '(repetition - 1) / 2'},
                {'index': 14, 'name': 'ply', 'formula': 'ply / 100'},
                {'index': 15, 'name': 'timed', 'formula': 'timed ? 1 : 0'},
            ],
            'notes': 'timed = clock is not NaN. log1p(x) = ln(1 + x). The training pipeline computes these in '
                     'float32; float64 then a Float32Array cast agrees within 1e-6. There is no clipping of any '
                     'feature. The time control (base, inc, time_class, log_base) is set even when the clocks are '
                     'NaN, but in practice clocks and time control are either both known or both unknown.',
        },
        'play': play_settings(stats, player, bullet),
        'archive_stats': stats,
    }


def play_settings(stats, player, bullet_rating):
    top = stats['moves_by_time_control'][0] if stats['moves_by_time_control'] else {}
    latest = stats['latest_rating']
    return {
        'training_data': f'{player}\'s moves by time control, from the archive: '
                         + ', '.join(f"{m['time_control']} {m['time_class']} {m['share']:.0%}"
                                     for m in stats['moves_by_time_control'][:5])
                         + '. Neither the clone\'s nor the general model\'s training data contains a single untimed '
                           'position: every row has real clocks, so timed = 1, time_class in {0, 1, 2}, and finite '
                           'clock features. Feeding NaN clocks / UNKNOWN_CLASS puts the net out of distribution; '
                           'do not do it.',
        'clone': {
            'time_control': top.get('time_control'), 'time_class': top.get('time_class'),
            'elo': latest.get(top.get('time_class'), {}).get('rating'),
            'opp_elo': latest.get(top.get('time_class'), {}).get('rating'),
            'temperature': 1.0,
            'why': f"{top.get('share', 0):.0%} of {player}'s moves are {top.get('time_control')} "
                   f"{top.get('time_class')}; the latest {top.get('time_class')} rating is "
                   f"{latest.get(top.get('time_class'), {}).get('rating')}. Opponents on chess.com are matched near "
                   "one's own rating, so opp_elo = elo keeps elo_diff ~ 0, the bulk of the training rows.",
        },
        'human': {
            'time_control': '600', 'time_class': 'rapid', 'elo': 700, 'opp_elo': 700, 'temperature': 1.0,
            'why': 'The general model was trained on bullet, blitz and rapid in equal game quotas per rating bucket; '
                   'low-rated players in the data are mostly rapid (10+0) and blitz, and a 10-minute clock is the '
                   'least sensitive to how fast the visitor actually plays. A game is bucketed by the average of '
                   'both ratings, so a 700 Jester vs a 700 opponent is squarely in the 600-799 rows.',
        },
        'clocks': {
            'rule': 'Simulate both clocks even though the website game is untimed. The model\'s own clock: subtract '
                    'expm1(log_time) from its last clock after each of its moves (clamp the spent time to '
                    '[0.1 s, 0.5 * remaining]), then add the increment. The visitor\'s clock: subtract the wall-clock '
                    'seconds they took, capped per move at 15 s for bullet and 60 s otherwise, then add the '
                    'increment. Never let either clock go below 1 s; the model must not be told it has flagged, '
                    'and a game that runs long should look like a long game with a low clock, not a NaN one.',
            'why': 'The clock features (log_clock, clock_frac and the opponent versions) are what let a bullet clone '
                   'speed up and blunder under pressure like the real player. A wall clock would drain a 60-second '
                   'bullet clock in two or three visitor moves, so the visitor\'s time per move is capped.',
        },
        'temperature': 'policy logits / T before the softmax over legal moves, then sample. T = 1 samples the '
                       'predicted human distribution (the README\'s "plays like a human"); T = 0 is the argmax and '
                       'plays the same deterministic game every time. Use 1.0.',
        'value_head': 'softmax(value) = P(loss), P(draw), P(win) for the side to move; useful for a resignation rule '
                      '(e.g. resign when P(win) < 0.02 for 3 consecutive moves after move 20) but not needed to play.',
    }


# --- CLI ----------------------------------------------------------------------
def export(models, out, archive, player, seed=0, fixtures=50, eval_positions=400, precisions=PRECISIONS,
           min_agreement=0.99, max_nll_drift=0.02):
    """Export `models` ({name: checkpoint}) into `out`; returns the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    games = load_archive(archive, player)
    if not games:
        raise SystemExit(f'no games in {archive}')
    golden = sample_positions(games, rng, player)[:fixtures]
    keys = {(g.uuid, p) for g, p, _ in golden}
    for game, ply in random_positions(games, rng, max(0, fixtures - len(golden)), player, keys):
        golden.append((game, ply, sorted(position_tags(rewind(game, ply).board))))
        keys.add((game.uuid, ply))
    eval_set = random_positions(games, rng, eval_positions, player, keys)
    loaded, ready = {}, {}
    for name, path in models.items():
        model, _ = net.load(path)
        fp32_path = out / f'{name}.fp32.onnx'
        export_fp32(model, fp32_path)
        onnx_path = out / f'{name}.onnx'
        precision, reports = choose_precision(model, fp32_path, onnx_path, eval_set, precisions,
                                              min_agreement, max_nll_drift)
        fp32_path.unlink()
        r = reports[precision]
        print(f'{name}: {precision} {r["bytes"] / 1e6:.2f} MB; over {len(eval_set)} positions argmax agreement '
              f'{r["argmax_agreement"]:.2%} ({r["argmax_agreement_clear"]:.2%} of the {r["clear_positions"]} with a '
              f'clear best move), nll drift {r["nll_drift"]:+.4f}, max prob diff {r["max_prob_diff"]:.4f}')
        loaded[name] = (model, path, precision, reports)
        ready[name] = (model, OnnxModel(onnx_path))
    stats = archive_stats(games, player)
    data = manifest(loaded, stats, player)
    (out / 'manifest.json').write_text(json.dumps(data, indent=1) + '\n')
    vectors = [fixture(game, ply, tags, ready) for game, ply, tags in golden]
    covered = collections.Counter(t for v in vectors for t in v['tags'])
    (out / 'fixtures.json').write_text(json.dumps({'contract_version': CONTRACT_VERSION,
                                                   'encoding_version': encoding.ENCODING_VERSION,
                                                   'tags': dict(covered), 'fixtures': vectors}, indent=1) + '\n')
    missing = [t for t in TAG_QUOTA if not covered[t]]
    print(f'{len(vectors)} fixtures; tags {dict(covered)}' + (f'; MISSING {missing}' if missing else ''))
    for p in sorted(out.iterdir()):
        print(f'{p.stat().st_size / 1e6:8.2f} MB  {p.name}')
    return data


def main(argv=None):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', action='append', required=True, metavar='NAME=CHECKPOINT')
    ap.add_argument('--out', required=True)
    ap.add_argument('--archive', default='data/players/hydra33092', help='monthly chess.com JSON for the fixtures')
    ap.add_argument('--player', default='hydra33092', help='fixture positions are this player\'s moves')
    ap.add_argument('--fixtures', type=int, default=50)
    ap.add_argument('--eval-positions', type=int, default=400, help='extra positions for the precision choice')
    ap.add_argument('--precision', choices=ALL_PRECISIONS, help='force one precision instead of trying int8, fp16, fp32')
    ap.add_argument('--min-agreement', type=float, default=0.99,
                    help='argmax agreement with torch on positions where torch has a clear best move')
    ap.add_argument('--max-nll-drift', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)
    models = dict(m.split('=', 1) for m in args.model)
    export(models, args.out, args.archive, args.player, args.seed, args.fixtures, args.eval_positions,
           (args.precision,) if args.precision else PRECISIONS, args.min_agreement, args.max_nll_drift)


if __name__ == '__main__':
    main()
