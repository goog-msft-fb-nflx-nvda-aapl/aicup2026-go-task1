"""Rank fine-tuning of the pretrained rank-conditioned move LM (lm checkpoint) with a discriminative head and ordinal-loss ablation.
usage: CUDA_VISIBLE_DEVICES=k python train_ft.py <tag> --init lm1 --loss ce|sord|emd|coral [--tau 1.0 --epochs 2]
Evaluates game exact + question-level sim on held-out players each epoch; saves hold per-game log-probs."""
import sys, time, math, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, (__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from train_lm import LM, sym_tables, D
ap = argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('--init', default='lm1'); ap.add_argument('--loss', default='ce'); ap.add_argument('--tau', type=float, default=1.0)
ap.add_argument('--epochs', type=int, default=2); ap.add_argument('--bs', type=int, default=128); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--maxlen', type=int, default=320); ap.add_argument('--pool', default='own')
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device('cuda', 0); t = torch.arange(32, device=dev, dtype=torch.float32); assert (t * t).sum().item() == 10416; print('REAL_TENSOR_PASS', flush=True)
ck = torch.load(D + f'results/lm_{a.init}.pt', map_location='cpu', weights_only=False); la = ck['args']; hold = ck['hold']
lm = LM(la['d'], la['layers'], la['heads']); lm.load_state_dict(ck['net'])
with torch.no_grad(): lm.cond.weight[10] = lm.cond.weight[:10].mean(0)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.lm = lm; d = la['d']; s.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1 if a.loss == 'coral' else 10)) if a.pool == 'own' else nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1 if a.loss == 'coral' else 10))
        if a.loss == 'coral': s.bias = nn.Parameter(torch.linspace(1.5, -1.5, 9))
    def forward(s, pos, col, hd, tgt):
        B, L = pos.shape; m = s.lm; ply = torch.arange(L, device=pos.device)[None].expand(B, L); cond = torch.full((B,), 10, device=pos.device)
        x = m.pos(pos) + m.col(col) + m.ply(ply) + (m.hd(hd) + m.tgt(tgt) + m.cond(cond))[:, None]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=pos.device)
        h = m.ln(m.enc(x, mask=mask, is_causal=True)); own = ((col == tgt[:, None]) & (pos != 362)).float().unsqueeze(-1)
        pooled = (h * own).sum(1) / own.sum(1).clamp(min=1)
        if a.pool == 'both':
            allm = (pos != 362).float().unsqueeze(-1); pooled = torch.cat([pooled, (h * allm).sum(1) / allm.sum(1).clamp(min=1)], 1)
        o = s.head(pooled.float())
        return o
    def logprobs(s, o):
        if a.loss == 'coral':
            cum = torch.sigmoid(o + s.bias[None])  # P(y > k) , k=0..8   (o: (B,1))
            p = torch.cat([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], 1).clamp(min=1e-6); return torch.log(p / p.sum(1, keepdim=True))
        return F.log_softmax(o, -1)
def loss_fn(net, o, y):
    if a.loss == 'ce': return F.cross_entropy(o, y, label_smoothing=0.05)
    if a.loss == 'sord':
        r = torch.arange(10, device=o.device)[None].float(); tgt = F.softmax(-(r - y[:, None].float()).abs() / a.tau, -1); return -(tgt * F.log_softmax(o, -1)).sum(1).mean()
    if a.loss == 'emd':
        p = F.softmax(o, -1); cp = p.cumsum(1); ct = (torch.arange(10, device=o.device)[None] >= y[:, None]).float(); return ((cp - ct) ** 2).sum(1).mean() + 0.1 * F.cross_entropy(o, y)
    if a.loss == 'coral':
        lv = (torch.arange(9, device=o.device)[None] < y[:, None]).float(); lg = o + net.bias[None]; return F.binary_cross_entropy_with_logits(lg, lv)
z = np.load(D + 'data/train_arr.npz'); M = z['M']; C = z['C']; hd = z['hd']; y = z['y']; col = z['col']; P = z['P']
ishold = np.isin(P, hold); tr = np.flatnonzero(~ishold); va = np.flatnonzero(ishold)
Mg = torch.from_numpy(M[:, :a.maxlen].astype(np.int64)).to(dev); Cg = torch.from_numpy(C[:, :a.maxlen].astype(np.int64)).to(dev); Mg[Mg == 65535] = 362
hdg = torch.from_numpy(np.minimum(hd, 9).astype(np.int64)).to(dev); yg = torch.from_numpy(y.astype(np.int64)).to(dev); colg = torch.from_numpy(col.astype(np.int64)).to(dev); SYM = sym_tables(dev)
net = Net().to(dev); params = [{'params': net.lm.parameters(), 'lr': a.lr}, {'params': list(net.head.parameters()) + ([net.bias] if a.loss == 'coral' else []), 'lr': 1e-3}]
opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.98)); steps_per = min(len(tr) // a.bs, int(__import__('os').environ.get('GO_MAX_STEPS','1000000000'))); total = steps_per * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, 1e-3], total_steps=total, pct_start=0.05); trg = torch.from_numpy(tr).to(dev)
byp = {}
for g in va: byp.setdefault((P[g], col[g]), []).append(g)
keys = [k for k, v in byp.items() if len(v) >= 20]; need = np.array(sorted({i for k in keys for i in byp[k]})); loc = {g: i for i, g in enumerate(need)}
def qs(lp, t):
    p = np.exp(lp - lp.max()); p /= p.sum(); sc = p.copy(); sc[1:] += math.exp(-1) * p[:-1]; sc[:-1] += math.exp(-1) * p[1:]; k = sc.argmax(); return 1. if k == t else (math.exp(-1) if abs(k - t) == 1 else 0.)
@torch.no_grad()
def evaluate(tta=1):
    net.eval(); LP = np.zeros((len(need), 10), np.float32)
    for s in range(0, len(need), 256):
        ib = torch.from_numpy(need[s:s + 256]).to(dev); acc = 0
        for k in range(tta):
            pos = torch.gather(SYM[k].expand(len(ib), -1), 1, Mg[ib])
            with torch.autocast('cuda', dtype=torch.bfloat16): o = net(pos, Cg[ib], hdg[ib], colg[ib])
            acc = acc + net.logprobs(o.float())
        LP[s:s + 256] = (acc / tta).cpu().numpy()
    net.train(); ex = (LP.argmax(1) == y[need]).mean(); r2 = np.random.default_rng(3); tot = []
    for n in (5, 10, 15, 20):
        sc = []
        for _ in range(600):
            k = keys[r2.integers(len(keys))]; ix = r2.choice(byp[k], n, replace=False); sc.append(qs(LP[[loc[i] for i in ix]].sum(0), y[ix[0]]))
        tot += sc
    return ex, float(np.mean(tot)), LP
t0 = time.time(); net.train()
for ep in range(a.epochs):
    perm = trg[torch.randperm(len(trg), device=dev)]; tl = 0
    for s in range(steps_per):
        ib = perm[s * a.bs:(s + 1) * a.bs]; k = torch.randint(0, 8, (len(ib),), device=dev); pos = torch.gather(SYM[k], 1, Mg[ib])
        with torch.autocast('cuda', dtype=torch.bfloat16): o = net(pos, Cg[ib], hdg[ib], colg[ib])
        loss = loss_fn(net, o.float(), yg[ib]); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item()
        if s % 1000 == 0: print(f'ep{ep} step{s}/{steps_per} loss {tl / (s + 1):.4f} t={time.time() - t0:.0f}s', flush=True)
    ex, qsim, LP = evaluate(); print(f'EPOCH {ep} {a.loss} loss {tl / steps_per:.4f} val_game_exact {ex:.4f} question_sim {qsim:.4f} t={time.time() - t0:.0f}s', flush=True)
    torch.save({'net': net.state_dict(), 'args': vars(a)}, D + f'results/ft_{a.tag}.pt'); np.save(D + f'results/ft_{a.tag}_holdLP.npy', LP); np.save(D + f'results/ft_{a.tag}_holdidx.npy', need)
print('DONE', flush=True)
