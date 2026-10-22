"""Play with a trained checkpoint: `model:path=runs/hydra/best.pt,temperature=1`.

temperature=0 plays the most likely move; 1 samples from the predicted human
distribution (more varied, closer to how a person actually plays).
"""
import chess
import numpy as np
import torch

from .. import encoding
from ..dataset.reader import default_device
from ..engine.players import Player
from . import net
from .inputs import game_inputs


class ModelPlayer(Player):
    def __init__(self, path, temperature=0.0, device=None, seed=None):
        self.device = torch.device(device) if device else default_device()
        self.model, _ = net.load(path, self.device)
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(None if seed is None else int(seed))
        self.name = f'model-{path}'

    @torch.no_grad()
    def predict(self, game):
        """Legal-masked move probabilities (1858,), win/draw/loss, expected seconds."""
        planes, feats = game_inputs(game, self.model.history)
        out = self.model(torch.from_numpy(planes).to(self.device), torch.from_numpy(feats).to(self.device))
        logits = out['policy'][0].cpu().numpy().astype(np.float64)
        logits[~encoding.legal_mask(game.board)] = -np.inf
        probs = np.exp(logits - logits.max())
        value = torch.softmax(out['value'][0], 0).cpu().numpy()
        return probs / probs.sum(), {'loss': value[0], 'draw': value[1], 'win': value[2]}, \
            float(torch.expm1(out['log_time'][0]))

    def choose_move(self, game):
        probs, _, _ = self.predict(game)
        if self.temperature <= 0:
            index = int(probs.argmax())
        else:
            p = probs ** (1 / self.temperature)
            index = int(self.rng.choice(len(p), p=p / p.sum()))
        return encoding.index_to_move(index, game.board)

    def top_moves(self, game, k=5):
        probs, _, _ = self.predict(game)
        best = np.argsort(-probs)[:k]
        return [(game.board.san(encoding.index_to_move(int(i), game.board)), float(probs[i])) for i in best]
