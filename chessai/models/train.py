"""Train or fine-tune the policy net, and evaluate checkpoints.

    # 1. general human model on the bulk dataset
    python -m chessai.models.train --data data/datasets/bulk --out runs/base \\
        --steps 100000 --in-memory --eval data/datasets/hydra

    # 2. fine-tune on your own moves
    python -m chessai.models.train --data data/datasets/hydra --target-only \\
        --init runs/base/best.pt --out runs/hydra --steps 20000 --lr 3e-4

    # evaluate on your held-out (newest) games
    python -m chessai.models.train --evaluate runs/hydra/best.pt --data data/datasets/hydra --split test

Evaluation sets use target rows when the dataset has target players, so the
headline number is "how often does the model play exactly the move you played".
"""
import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..dataset.reader import PositionDataset, default_device
from ..dataset.schema import TIME_CLASSES
from . import net

TIME_LOSS, VALUE_LOSS = 0.1, 0.25


def to_device(batch, device):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in batch.items()}


def prefetch(iterator, depth=4):
    """Build numpy batches on a background thread while the GPU trains.

    An exception in the reader thread is re-raised here, so training fails
    instead of waiting forever for the next batch.
    """
    q = queue.Queue(depth)
    done = object()

    def run():
        try:
            for item in iterator:
                q.put(item)
            q.put(done)
        except BaseException as e:
            q.put(e)
    threading.Thread(target=run, daemon=True).start()
    while (item := q.get()) is not done:
        if isinstance(item, BaseException):
            raise item
        yield item


def infinite_batches(ds, idx, batch_size, history, seed=0):
    epoch = 0
    while True:
        yield from ds.batches(idx, batch_size, shuffle=True, seed=seed + epoch, history=history, drop_last=True)
        epoch += 1


def losses(out, batch):
    logits = out['policy']
    if 'legal' in batch:
        logits = logits.masked_fill(~batch['legal'], -1e9)
    policy = F.cross_entropy(logits, batch['move'])
    value = F.cross_entropy(out['value'], (batch['result'] + 1).long())
    timed = ~torch.isnan(batch['move_time'])
    time_loss = F.mse_loss(out['log_time'][timed], torch.log1p(batch['move_time'][timed])) \
        if timed.any() else torch.zeros((), device=logits.device)
    total = policy + VALUE_LOSS * value + TIME_LOSS * time_loss
    return total, {'policy': policy.item(), 'value': value.item(), 'time': time_loss.item(),
                   'acc': (logits.argmax(1) == batch['move']).float().mean().item()}


@torch.no_grad()
def evaluate(model, ds, idx, device, batch_size=2048):
    """Move-match accuracy (top-1/top-3), NLL, and per-time-class top-1."""
    model.eval()
    n = top1 = top3 = nll = 0
    per_class = {}
    for b in ds.batches(idx, batch_size, history=model.history):
        tc = b['features'][:, 9:12].argmax(1)  # bullet/blitz/rapid one-hot
        has_tc = b['features'][:, 9:12].any(1)
        b = to_device(b, device)
        logits = model(b['planes'], b['features'])['policy']
        if 'legal' in b:
            logits = logits.masked_fill(~b['legal'], -1e9)
        hit = (logits.argmax(1) == b['move']).cpu().numpy()
        top1 += hit.sum()
        top3 += (logits.topk(3, 1).indices == b['move'][:, None]).any(1).sum().item()
        nll += F.cross_entropy(logits, b['move'], reduction='sum').item()
        n += len(hit)
        for c, name in enumerate(TIME_CLASSES):
            sel = has_tc & (tc == c)
            if sel.any():
                h, t = per_class.get(name, (0, 0))
                per_class[name] = (h + int(hit[sel].sum()), t + int(sel.sum()))
    model.train()
    return {'rows': n, 'top1': top1 / max(n, 1), 'top3': top3 / max(n, 1), 'nll': nll / max(n, 1),
            **{f'top1_{k}': h / t for k, (h, t) in per_class.items()}}


def eval_indices(ds, split, limit, seed=0):
    target = True if ds.manifest.get('target_rows') else None
    idx = ds.select(split, target=target)
    if limit and len(idx) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(idx, limit, replace=False))
    return idx


def eval_names(paths):
    """{resolved path: log name}: the folder name, or the full path if two
    different datasets share a folder name. Repeated paths appear once."""
    resolved = list(dict.fromkeys(Path(p).resolve() for p in paths))
    counts = {}
    for path in resolved:
        counts[path.name] = counts.get(path.name, 0) + 1
    return {path: path.name if counts[path.name] == 1 else str(path) for path in resolved}


def time_class_filter(name):
    if not name:
        return None
    c = TIME_CLASSES.index(name)
    return lambda rows: rows['time_class'] == c


def train(args):
    device = torch.device(args.device) if args.device else default_device()
    ds = PositionDataset(args.data, in_memory=args.in_memory)
    train_idx = ds.select('train', target=True if args.target_only else None,
                          where=time_class_filter(args.time_class))
    names = eval_names([args.data] + args.eval)
    evals = {}
    for path, name in names.items():
        eds = ds if path == Path(args.data).resolve() else PositionDataset(path)
        evals[name] = (eds, eval_indices(eds, 'val', args.eval_rows))
    primary = names[Path(args.data).resolve()]

    if args.init:
        model, _ = net.load(args.init, device)
        model.train()
    else:
        model = net.PolicyNet(history=args.history, blocks=args.blocks, channels=args.channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(1000, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'log.jsonl', 'a')
    params = sum(p.numel() for p in model.parameters())
    print(f'{len(train_idx):,} training rows on {device}; {params:,} parameters; config {model.config}')

    best, started, running = -1.0, time.time(), {}
    batches = prefetch(infinite_batches(ds, train_idx, args.batch, model.history, args.seed))
    for step in range(1, args.steps + 1):
        batch = to_device(next(batches), device)
        loss, stats = losses(model(batch['planes'], batch['features']), batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        for k, v in stats.items():
            running[k] = running.get(k, 0) + v
        if step % args.log_every == 0:
            rate = step * args.batch / (time.time() - started)
            print(f'step {step} ' + ' '.join(f'{k} {v / args.log_every:.4f}' for k, v in running.items())
                  + f' lr {sched.get_last_lr()[0]:.2e} {rate:,.0f} pos/s', flush=True)
            running = {}
        if step % args.eval_every == 0 or step == args.steps:
            record = {'step': step, 'time': round(time.time() - started)}
            for name, (eds, idx) in evals.items():
                record[name] = evaluate(model, eds, idx, device)
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + '\n')
            log.flush()
            score = record[primary]['top1']
            net.save(model, out / 'last.pt', step=step, args=vars(args))
            if score > best:
                best = score
                net.save(model, out / 'best.pt', step=step, args=vars(args), val=record)
    print(f'best val top-1 {best:.4f} -> {out / "best.pt"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out')
    ap.add_argument('--evaluate', metavar='CHECKPOINT', help='evaluate instead of training')
    ap.add_argument('--split', default='test', help='--evaluate split')
    ap.add_argument('--eval', action='append', default=[], help='extra dataset to validate on')
    ap.add_argument('--eval-rows', type=int, default=50_000)
    ap.add_argument('--init', help='checkpoint to fine-tune from')
    ap.add_argument('--target-only', action='store_true', help="train only on the target player's moves")
    ap.add_argument('--time-class', choices=TIME_CLASSES)
    ap.add_argument('--history', type=int, default=2)
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--steps', type=int, default=20_000)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--in-memory', action='store_true')
    ap.add_argument('--device')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.evaluate:
        device = torch.device(args.device) if args.device else default_device()
        model, _ = net.load(args.evaluate, device)
        ds = PositionDataset(args.data)
        print(json.dumps(evaluate(model, ds, eval_indices(ds, args.split, None), device), indent=1))
    elif not args.out:
        ap.error('--out is required for training')
    else:
        train(args)


if __name__ == '__main__':
    main()
