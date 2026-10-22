"""Policy network: predicts the human move, the game result, and the time spent.

A Maia-style residual conv net. Scalar features (ratings, clocks, time
control, ...) are broadcast as constant planes into the stem and fed again to
the time head, so the move distribution can depend on who is playing and how
much time they have.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import encoding
from ..dataset.reader import NUM_FEATURES

BOARD_PLANES = 17
HISTORY_PLANES = 12


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(y)))


class PolicyNet(nn.Module):
    def __init__(self, history=2, blocks=6, channels=64, num_features=NUM_FEATURES):
        super().__init__()
        self.config = dict(history=history, blocks=blocks, channels=channels, num_features=num_features)
        in_planes = BOARD_PLANES + HISTORY_PLANES * history + num_features
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(32 * 64, encoding.NUM_MOVES))
        self.trunk_small = nn.Sequential(nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
                                         nn.Flatten())
        self.value = nn.Sequential(nn.Linear(8 * 64 + num_features, 128), nn.ReLU(), nn.Linear(128, 3))
        self.time = nn.Sequential(nn.Linear(8 * 64 + num_features, 64), nn.ReLU(), nn.Linear(64, 1))

    @property
    def history(self):
        return self.config['history']

    def forward(self, planes, features):
        """planes (B, 17+12h, 8, 8), features (B, F) ->
        policy logits (B, 1858), value logits (B, 3: loss/draw/win), log1p seconds (B,)."""
        b = planes.shape[0]
        x = torch.cat([planes, features[:, :, None, None].expand(b, -1, 8, 8)], 1)
        x = self.tower(self.stem(x))
        small = torch.cat([self.trunk_small(x), features], 1)
        return {'policy': self.policy(x), 'value': self.value(small), 'log_time': self.time(small)[:, 0]}


def save(model, path, **extra):
    torch.save({'config': model.config, 'state_dict': model.state_dict(),
                'encoding_version': encoding.ENCODING_VERSION, **extra}, path)


def load(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt['encoding_version'] != encoding.ENCODING_VERSION:
        raise ValueError(f'{path} was trained with encoding v{ckpt["encoding_version"]}')
    model = PolicyNet(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval(), ckpt
