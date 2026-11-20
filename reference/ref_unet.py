"""Reference implementation used only to verify numbers quoted in the plan docs."""
import math, torch, torch.nn as nn, torch.nn.functional as F

def sinusoidal(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)

class ResBlock(nn.Module):
    def __init__(s, cin, cout, temb=256, groups=8, p=0.1):
        super().__init__()
        s.n1, s.c1 = nn.GroupNorm(groups, cin), nn.Conv2d(cin, cout, 3, padding=1)
        s.t = nn.Linear(temb, 2 * cout)
        s.n2, s.d, s.c2 = nn.GroupNorm(groups, cout), nn.Dropout(p), nn.Conv2d(cout, cout, 3, padding=1)
        s.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(s, x, te):
        h = s.c1(F.silu(s.n1(x)))
        sc, sh = s.t(F.silu(te))[:, :, None, None].chunk(2, 1)
        h = s.n2(h) * (1 + sc) + sh
        h = s.c2(s.d(F.silu(h)))
        return h + s.skip(x)

class Attn(nn.Module):
    def __init__(s, c, ctx=None, heads=4, groups=8):
        super().__init__()
        s.n = nn.GroupNorm(groups, c); s.h = heads
        s.q = nn.Linear(c, c); s.k = nn.Linear(ctx or c, c); s.v = nn.Linear(ctx or c, c); s.o = nn.Linear(c, c)
    def forward(s, x, ctx=None, mask=None):
        B, C, H, W = x.shape
        h = s.n(x).flatten(2).transpose(1, 2)
        kv = h if ctx is None else ctx
        q = s.q(h).view(B, -1, s.h, C // s.h).transpose(1, 2)
        k = s.k(kv).view(B, -1, s.h, C // s.h).transpose(1, 2)
        v = s.v(kv).view(B, -1, s.h, C // s.h).transpose(1, 2)
        am = None if mask is None else mask[:, None, None, :]  # True = attend
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        o = s.o(o.transpose(1, 2).reshape(B, H * W, C)).transpose(1, 2).view(B, C, H, W)
        return x + o

class UNet(nn.Module):
    def __init__(s, cin=13, base=64, mult=(1, 2, 4), ctx=128):
        super().__init__()
        ch = [base * m for m in mult]
        s.tmlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 256))
        s.inc = nn.Conv2d(cin, base, 3, padding=1)
        s.down = nn.ModuleList(); skips = [base]; c = base
        for i, co in enumerate(ch):
            for _ in range(2):
                blk = nn.ModuleList([ResBlock(c, co)]); c = co
                if i >= 1: blk.append(Attn(c)); blk.append(Attn(c, ctx))
                s.down.append(blk); skips.append(c)
            if i < len(ch) - 1:
                s.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)])); skips.append(c)
        s.mid = nn.ModuleList([ResBlock(c, c), Attn(c), Attn(c, ctx), ResBlock(c, c)])
        s.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(ch))):
            for _ in range(3):
                blk = nn.ModuleList([ResBlock(c + skips.pop(), co)]); c = co
                if i >= 1: blk.append(Attn(c)); blk.append(Attn(c, ctx))
                s.up.append(blk)
            if i > 0:
                s.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
        s.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, cin, 3, padding=1))
    def forward(s, x, t, ctx, mask=None):
        te = s.tmlp(sinusoidal(t)); h = s.inc(x); hs = [h]
        for blk in s.down:
            if isinstance(blk[0], nn.Conv2d): h = blk[0](h)
            else:
                h = blk[0](h, te)
                if len(blk) > 1: h = blk[1](h); h = blk[2](h, ctx, mask)
            hs.append(h)
        h = s.mid[0](h, te); h = s.mid[1](h); h = s.mid[2](h, ctx, mask); h = s.mid[3](h, te)
        for blk in s.up:
            if isinstance(blk[0], nn.Upsample): h = blk[1](blk[0](h))
            else:
                h = blk[0](torch.cat([h, hs.pop()], 1), te)
                if len(blk) > 1: h = blk[1](h); h = blk[2](h, ctx, mask)
        return s.out(h)

if __name__ == "__main__":
    m = UNet(); n = sum(p.numel() for p in m.parameters())
    adapter = nn.Sequential(nn.Linear(384, 128), nn.SiLU(), nn.Linear(128, 128), nn.LayerNorm(128))
    na = sum(p.numel() for p in adapter.parameters())
    print("unet params", n, "adapter", na)
    x = torch.randn(4, 13, 16, 16); t = torch.randint(0, 1000, (4,)); c = torch.randn(4, 20, 128)
    mask = torch.ones(4, 20, dtype=torch.bool); mask[:, 15:] = False
    print("out", m(x, t, c, mask).shape)
    xw = torch.randn(2, 13, 16, 112); print("wide out", m(xw, t[:2], c[:2], mask[:2]).shape)
    # per-stage param breakdown
    for name, mod in [("tmlp", m.tmlp), ("down", m.down), ("mid", m.mid), ("up", m.up), ("out", m.out)]:
        print(name, sum(p.numel() for p in mod.parameters()))

    # --- scheduler sanity ---
    T = 1000
    def linear(): return torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    def cosine(s=0.008):
        st = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((st + s) / (1 + s) * math.pi / 2) ** 2
        ab = f / f[0]; b = 1 - ab[1:] / ab[:-1]; return b.clamp(max=0.999)
    for nm, b in [("linear", linear()), ("cosine", cosine())]:
        ab = torch.cumprod(1 - b, 0)
        print(nm, "abar_T", ab[-1].item(), "abar_500", ab[499].item(), "SNR_mid", (ab[499] / (1 - ab[499])).item())
        # oracle DDIM reconstruction
        x0 = (F.one_hot(torch.randint(0, 13, (8, 16, 16)), 13).permute(0, 3, 1, 2).double() * 2 - 1)
        ts = torch.linspace(T - 1, 0, 50).round().long()
        x = torch.randn_like(x0)
        for i, t in enumerate(ts):
            a = ab[t]; ap = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, dtype=torch.float64)
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()   # oracle
            x0h = (x - (1 - a).sqrt() * eps) / a.sqrt()
            x = ap.sqrt() * x0h + (1 - ap).sqrt() * eps
        print(nm, "oracle DDIM mse", F.mse_loss(x, x0).item(), "argmax agree", (x.argmax(1) == x0.argmax(1)).float().mean().item())
        xt = ab[-1].sqrt() * x0 + (1 - ab[-1]).sqrt() * torch.randn_like(x0)
        print(nm, "x_999 mean/std", xt.mean().item(), xt.std().item())
    # memory estimate at batch 64 (CPU proxy: count activations is hard; report param bytes)
    print("param MB fp32", n * 4 / 1e6, "train-state MB (w+ema+grad+2 adam)", n * 4 * 5 / 1e6)
